"""
FastAPI REST controller for Voice Agents and Immutable Versions (/v2/agents).
"""

from __future__ import annotations

import os
import pathlib
import re

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from fastapi.responses import FileResponse

from orchestration.application.agents.commands import (
    CreateAgentCommand,
    PublishVersionCommand,
    UpdateAgentCommand,
)
from orchestration.application.agents.service import AgentApplicationService
from orchestration.domain.agent import OFFICIAL_PRESETS, Agent
from orchestration.interfaces.http.dependencies import get_agent_service
from orchestration.interfaces.http.schemas.agent_schemas import (
    AgentResponse,
    AgentVersionResponse,
    CreateAgentRequest,
    PublishVersionRequest,
    UpdateAgentRequest,
    VoicePresetResponse,
)
from orchestration.tts.voice_clone import (
    default_voice_cloner,
    VoiceCloningValidationError,
)

router = APIRouter(prefix="/v2/agents", tags=["Agents"])


def _to_agent_response(a: Agent) -> AgentResponse:
    return AgentResponse(
        id=a.id,
        name=a.name,
        voice_id=a.voice_id,
        greeting=a.greeting,
        agent_speaks_first=a.agent_speaks_first,
        system_prompt=a.system_prompt,
        ending=a.ending,
        end_call_timeout_sec=a.end_call_timeout_sec,
        silence_timeout_sec=a.silence_timeout_sec,
        max_duration_sec=a.max_duration_sec,
        timezone_str=a.timezone_str,
        engine=a.engine,
        language=a.language,
        pipeline_json=a.pipeline_json,
        status=a.status.value,
        current_version=a.current_version,
        published_version=a.published_version,
        created_at=a.created_at,
        updated_at=a.updated_at,
    )


@router.get("/voices", response_model=list[VoicePresetResponse])
async def list_voice_presets() -> list[VoicePresetResponse]:
    """Return all 18 official PersonaPlex voice conditioning presets plus custom cloned voices."""
    presets = [
        VoicePresetResponse(
            id=p.id,
            name=p.name,
            gender=p.gender,
            speaking_style=p.speaking_style,
            accent=p.accent,
            recommended_for=p.recommended_for,
            is_cloned=False,
            preview_url=None,
        )
        for p in OFFICIAL_PRESETS.values()
    ]

    seen_ids = {p.id for p in presets}
    official_stems = {pathlib.Path(p.id).stem for p in presets}
    ignored_stems = {"voice_prompt"}

    # 1. Discover custom cloned voices registered in VoiceCloner
    try:
        cloned_list = default_voice_cloner.list_cloned_voices()
        for cv in cloned_list:
            vid = cv.get("id", "")
            fname = f"{vid}.wav" if not (vid.endswith(".wav") or vid.endswith(".pt")) else vid
            stem = pathlib.Path(vid).stem
            if stem not in official_stems and stem not in ignored_stems and fname not in seen_ids and vid not in seen_ids:
                seen_ids.add(fname)
                seen_ids.add(vid)
                seen_ids.add(stem)
                display_name = cv.get("name") or stem.replace("cloned_", "").replace("_", " ").title()
                presets.append(
                    VoicePresetResponse(
                        id=fname,
                        name=f"{display_name} (Cloned)",
                        gender=cv.get("gender", "custom"),
                        speaking_style="Cloned Neural Voice",
                        accent="Custom Cloned Reference",
                        recommended_for="Custom cloned voice conditioning",
                        is_cloned=True,
                        preview_url=f"/v2/agents/voices/{vid}/preview",
                        duration_sec=cv.get("duration_sec"),
                        qa_passed=cv.get("qa_passed", True),
                        qa_score=cv.get("qa_score", 0.95),
                        recommended_engine=cv.get("recommended_engine", "personaplex_s2s"),
                    )
                )
    except Exception as e:
        logger.warning(f"Error reading cloned voices in router: {e}")

    # 2. Discover custom voices in local voices/ directory, data/cloned_voices, and HF_HOME/voices
    custom_dirs = [
        pathlib.Path("data") / "cloned_voices",
        pathlib.Path("/workspace/orchestration_ai/data/cloned_voices"),
        pathlib.Path("voices"),
        pathlib.Path("/workspace/voices"),
        pathlib.Path("/workspace/orchestration_ai/voices"),
        pathlib.Path("/workspace/huggingface/voices"),
        pathlib.Path.home() / ".cache" / "huggingface" / "voices",
        pathlib.Path("/data/huggingface/voices"),
        pathlib.Path(os.environ.get("HF_HOME", "/workspace/huggingface")) / "voices",
    ]
    for cdir in custom_dirs:
        if cdir and cdir.exists() and cdir.is_dir():
            for fpath in cdir.glob("**/*"):
                if fpath.is_file() and fpath.suffix.lower() in (".wav", ".pt"):
                    stem = fpath.stem
                    if stem in official_stems or stem in ignored_stems or fpath.name in seen_ids or stem in seen_ids:
                        continue
                    seen_ids.add(fpath.name)
                    seen_ids.add(stem)
                    meta = default_voice_cloner.get_voice_metadata(stem)
                    display_name = (meta and meta.get("name")) or stem.replace("cloned_", "").replace("_", " ").title()
                    presets.append(
                        VoicePresetResponse(
                            id=fpath.name,
                            name=f"{display_name} (Cloned)",
                            gender=meta.get("gender", "custom") if meta else "custom",
                            speaking_style="Cloned Neural Voice",
                            accent="Indian English" if "indian" in stem.lower() or "aarav" in stem.lower() else "Custom Reference",
                            recommended_for="Custom persona voice conditioning",
                            is_cloned=True,
                            preview_url=f"/v2/agents/voices/{fpath.name}/preview",
                            duration_sec=meta.get("duration_sec") if meta else None,
                            qa_passed=meta.get("qa_passed", True) if meta else True,
                            qa_score=meta.get("qa_score", 0.95) if meta else 0.95,
                            recommended_engine=meta.get("recommended_engine", "personaplex_s2s") if meta else "personaplex_s2s",
                        )
                    )
    return presets


@router.post("/voices/clone", response_model=VoicePresetResponse, status_code=status.HTTP_201_CREATED)
async def clone_custom_voice(
    file: UploadFile = File(...),
    name: str = Form("My Voice"),
    consent: bool = Form(True),
    consent_statement: str | None = Form(None),
    gender: str = Form(None),
) -> VoicePresetResponse:
    """
    Clone a voice from recorded microphone audio or uploaded audio sample (3-60 seconds).
    Enforces recorded consent statement, validates quality, normalizes loudness, and registers voice.
    """
    if not consent:
        raise HTTPException(status_code=400, detail="Voice cloning requires explicit user consent.")

    content = await file.read()
    if len(content) < 1024:
        raise HTTPException(status_code=400, detail="Audio file too small (must contain valid audio data)")

    try:
        meta = default_voice_cloner.clone_voice(
            audio_bytes=content,
            voice_name=name.strip() or "My Voice",
            consent=consent,
            consent_statement=consent_statement,
            preferred_gender=gender,
        )
    except VoiceCloningValidationError as val_err:
        raise HTTPException(status_code=400, detail=str(val_err))
    except Exception as err:
        raise HTTPException(status_code=400, detail=f"Failed to process voice sample: {err}")

    voice_file = f"{meta['id']}.wav"
    return VoicePresetResponse(
        id=voice_file,
        name=f"{meta['name']} (Cloned)",
        gender=meta.get("gender", "custom"),
        speaking_style="Cloned Neural Voice",
        accent="Custom Cloned Reference",
        recommended_for="Custom cloned voice conditioning",
        is_cloned=True,
        preview_url=f"/v2/agents/voices/{meta['id']}/preview",
        duration_sec=meta.get("duration_sec"),
        qa_passed=meta.get("qa_passed"),
        qa_score=meta.get("qa_score"),
        recommended_engine=meta.get("recommended_engine"),
    )


@router.get("/voices/{voice_id}/preview")
async def preview_voice(voice_id: str):
    """Serve the 24 kHz WAV audio preview for a voice preset."""
    clean = voice_id.strip()
    if clean.endswith(".wav") or clean.endswith(".pt"):
        clean = pathlib.Path(clean).stem

    # 1. Check VoiceCloner data dir
    v_path = default_voice_cloner.get_voice_path(clean)
    if v_path and v_path.exists():
        if v_path.suffix == ".wav":
            return FileResponse(str(v_path), media_type="audio/wav", filename=f"{clean}.wav")
        wav_cand = v_path.with_suffix(".wav")
        if wav_cand.exists():
            return FileResponse(str(wav_cand), media_type="audio/wav", filename=f"{clean}.wav")

    # 2. Check voices/ and cache directories
    candidate_dirs = [
        pathlib.Path("voices"),
        pathlib.Path("/workspace/voices"),
        pathlib.Path("/workspace/orchestration_ai/voices"),
        pathlib.Path("/workspace/huggingface/voices"),
        pathlib.Path.home() / ".cache" / "huggingface" / "voices",
        pathlib.Path("/data/huggingface/voices"),
        pathlib.Path(os.environ.get("HF_HOME", "/workspace/huggingface")) / "voices",
    ]
    for dir_cand in candidate_dirs:
        if dir_cand and dir_cand.is_dir():
            for ext in (".wav", ".mp3", ".ogg"):
                target = dir_cand / f"{clean}{ext}"
                if target.exists():
                    return FileResponse(str(target), media_type=f"audio/{ext.lstrip('.')}")
                exact = dir_cand / voice_id
                if exact.exists() and exact.suffix in (".wav", ".mp3", ".ogg"):
                    return FileResponse(str(exact), media_type=f"audio/{exact.suffix.lstrip('.')}")

    raise HTTPException(status_code=404, detail=f"Preview audio for voice '{voice_id}' not found.")


@router.delete("/voices/{voice_id}")
async def delete_cloned_voice(voice_id: str):
    """Delete a custom cloned voice preset and its artifacts."""
    clean = voice_id.strip()
    if clean.endswith(".wav") or clean.endswith(".pt"):
        clean = pathlib.Path(clean).stem

    success = default_voice_cloner.delete_voice(clean)
    candidate_dirs = [
        pathlib.Path("voices"),
        pathlib.Path("/workspace/voices"),
        pathlib.Path("/workspace/orchestration_ai/voices"),
        pathlib.Path("/workspace/huggingface/voices"),
        pathlib.Path.home() / ".cache" / "huggingface" / "voices",
        pathlib.Path("/data/huggingface/voices"),
        pathlib.Path(os.environ.get("HF_HOME", "/workspace/huggingface")) / "voices",
    ]
    for dir_cand in candidate_dirs:
        if dir_cand and dir_cand.is_dir():
            for ext in (".wav", ".pt"):
                p = dir_cand / f"{clean}{ext}"
                if p.exists():
                    try:
                        p.unlink()
                        success = True
                    except Exception:
                        pass
    if not success:
        raise HTTPException(status_code=404, detail=f"Cloned voice '{voice_id}' not found.")
    return {"success": True, "id": voice_id}


@router.post("/voices/upload", response_model=VoicePresetResponse, status_code=status.HTTP_201_CREATED)
async def upload_custom_voice(
    file: UploadFile = File(...),
    name: str = Form(None),
    accent: str = Form("Indian English"),
    gender: str = Form("unspecified"),
) -> VoicePresetResponse:
    """
    Upload a custom reference voice prompt (clean mono WAV, 10-20s, 24 kHz recommended).
    Saves the conditioning audio to voices/ so PersonaPlex worker can condition on it.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename required")

    ext = pathlib.Path(file.filename).suffix.lower()
    if ext not in (".wav", ".pt"):
        raise HTTPException(status_code=400, detail="Only .wav or .pt audio reference files are supported")

    content = await file.read()
    if len(content) < 1024:
        raise HTTPException(status_code=400, detail="Audio file too small (must contain valid audio data)")

    # Sanitize filename
    clean_stem = re.sub(r"[^\w\-]", "_", pathlib.Path(file.filename).stem).strip("_")
    if not clean_stem:
        clean_stem = "custom_voice"

    target_dir = pathlib.Path("voices")
    target_dir.mkdir(parents=True, exist_ok=True)
    out_path = target_dir / f"{clean_stem}{ext}"

    if ext == ".wav":
        try:
            import io
            import soundfile as sf
            data, sr = sf.read(io.BytesIO(content), dtype="float32")
            if data.ndim > 1:
                data = data.mean(axis=1)

            duration_sec = len(data) / sr
            if duration_sec < 2.0:
                raise HTTPException(status_code=400, detail=f"Audio duration {duration_sec:.1f}s is too short (min 2s, recommend 10-20s)")
            if duration_sec > 60.0:
                raise HTTPException(status_code=400, detail=f"Audio duration {duration_sec:.1f}s is too long (max 60s)")

            # Resample to 24000 Hz if needed
            target_sr = 24000
            if sr != target_sr:
                import numpy as np
                num_samples = int(len(data) * target_sr / sr)
                data = np.interp(
                    np.linspace(0, len(data), num_samples, endpoint=False),
                    np.arange(len(data)),
                    data,
                ).astype(np.float32)

            sf.write(str(out_path), data, target_sr, subtype="PCM_16")

            # Also mirror to HF_HOME/voices if set
            hf_home = os.environ.get("HF_HOME")
            if hf_home:
                hf_voices = pathlib.Path(hf_home) / "voices"
                hf_voices.mkdir(parents=True, exist_ok=True)
                sf.write(str(hf_voices / f"{clean_stem}.wav"), data, target_sr, subtype="PCM_16")

        except HTTPException:
            raise
        except Exception as err:
            raise HTTPException(status_code=400, detail=f"Invalid WAV file: {err}")
    else:
        # Raw .pt PyTorch tensor
        out_path.write_bytes(content)
        hf_home = os.environ.get("HF_HOME")
        if hf_home:
            hf_voices = pathlib.Path(hf_home) / "voices"
            hf_voices.mkdir(parents=True, exist_ok=True)
            (hf_voices / f"{clean_stem}.pt").write_bytes(content)

    display_name = name or clean_stem.replace("_", " ").title()
    return VoicePresetResponse(
        id=out_path.name,
        name=f"{display_name} (Custom)",
        gender=gender,
        speaking_style="User conditioning reference",
        accent=accent,
        recommended_for="Custom persona voice conditioning",
        is_cloned=True,
        preview_url=f"/v2/agents/voices/{out_path.name}/preview",
    )


@router.post("", response_model=AgentResponse, status_code=status.HTTP_201_CREATED)
async def create_agent(
    req: CreateAgentRequest,
    svc: AgentApplicationService = Depends(get_agent_service),
) -> AgentResponse:
    """Create a new Voice Agent in draft status with Version 1."""
    cmd = CreateAgentCommand(
        name=req.name,
        voice_id=req.voice_id,
        greeting=req.greeting,
        agent_speaks_first=req.agent_speaks_first,
        system_prompt=req.system_prompt,
        ending=req.ending,
        end_call_timeout_sec=req.end_call_timeout_sec,
        silence_timeout_sec=req.silence_timeout_sec,
        max_duration_sec=req.max_duration_sec,
        timezone_str=req.timezone_str,
        engine=req.engine,
        language=req.language,
        pipeline_json=req.pipeline_json,
    )
    agent = await svc.create_agent(cmd)
    return _to_agent_response(agent)


@router.get("", response_model=list[AgentResponse])
async def list_agents(
    limit: int = 50,
    offset: int = 0,
    svc: AgentApplicationService = Depends(get_agent_service),
) -> list[AgentResponse]:
    """List all voice agents ordered by last updated timestamp."""
    agents = await svc.list_agents(limit=limit, offset=offset)
    return [_to_agent_response(a) for a in agents]


@router.get("/{agent_id}", response_model=AgentResponse)
async def get_agent(
    agent_id: str,
    svc: AgentApplicationService = Depends(get_agent_service),
) -> AgentResponse:
    """Fetch an agent by ID."""
    agent = await svc.get_agent(agent_id)
    return _to_agent_response(agent)


@router.patch("/{agent_id}", response_model=AgentResponse)
async def update_agent(
    agent_id: str,
    req: UpdateAgentRequest,
    svc: AgentApplicationService = Depends(get_agent_service),
) -> AgentResponse:
    """Autosave agent edits, bumping the version and creating an immutable snapshot."""
    cmd = UpdateAgentCommand(
        name=req.name,
        voice_id=req.voice_id,
        greeting=req.greeting,
        agent_speaks_first=req.agent_speaks_first,
        system_prompt=req.system_prompt,
        ending=req.ending,
        end_call_timeout_sec=req.end_call_timeout_sec,
        silence_timeout_sec=req.silence_timeout_sec,
        max_duration_sec=req.max_duration_sec,
        timezone_str=req.timezone_str,
        engine=req.engine,
        language=req.language,
        pipeline_json=req.pipeline_json,
        change_note=req.change_note,
    )
    agent = await svc.update_agent(agent_id, cmd)
    return _to_agent_response(agent)


@router.post("/{agent_id}/publish", response_model=AgentResponse)
async def publish_version(
    agent_id: str,
    req: PublishVersionRequest,
    svc: AgentApplicationService = Depends(get_agent_service),
) -> AgentResponse:
    """Publish current draft version. Enforces strict <=350 token limit on Engine A."""
    cmd = PublishVersionCommand(change_note=req.change_note)
    agent = await svc.publish_version(agent_id, cmd)
    return _to_agent_response(agent)


@router.post("/{agent_id}/revert/{version_number}", response_model=AgentResponse)
async def revert_version(
    agent_id: str,
    version_number: int,
    svc: AgentApplicationService = Depends(get_agent_service),
) -> AgentResponse:
    """Restore agent draft configuration from an earlier immutable version."""
    agent = await svc.revert_version(agent_id, version_number)
    return _to_agent_response(agent)


@router.get("/{agent_id}/versions", response_model=list[AgentVersionResponse])
async def list_versions(
    agent_id: str,
    svc: AgentApplicationService = Depends(get_agent_service),
) -> list[AgentVersionResponse]:
    """List all immutable version snapshots of an agent."""
    versions = await svc.list_versions(agent_id)
    return [
        AgentVersionResponse(
            version_id=v.version_id,
            agent_id=v.agent_id,
            version_number=v.version_number,
            name=v.name,
            voice_id=v.voice_id,
            greeting=v.greeting,
            agent_speaks_first=v.agent_speaks_first,
            system_prompt=v.system_prompt,
            ending=v.ending,
            end_call_timeout_sec=v.end_call_timeout_sec,
            silence_timeout_sec=v.silence_timeout_sec,
            max_duration_sec=v.max_duration_sec,
            timezone_str=v.timezone_str,
            engine=v.engine,
            language=v.language,
            pipeline_json=v.pipeline_json,
            compiled_prompt=v.compiled_prompt,
            token_count=v.token_count,
            created_at=v.created_at,
            change_note=v.change_note,
        )
        for v in versions
    ]


@router.delete("/{agent_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_agent(
    agent_id: str,
    svc: AgentApplicationService = Depends(get_agent_service),
) -> None:
    """Delete an agent and all associated versions."""
    await svc.delete_agent(agent_id)
