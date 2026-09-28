import asyncio
import os
import numpy as np
import pytest
import soundfile as sf
import uvicorn

from orchestration.gateway.app import create_app
from orchestration.worker.mock_worker import PersonaPlexMockServer
from orchestration.worker.pool import WorkerPool, WorkerNodeConfig
from orchestration.cli.main import _test_call_cmd
from orchestration.protocol.audio import SAMPLE_RATE


class MockArgs:
    def __init__(self, url, persona, input_wav, output_wav, output_json):
        self.url = url
        self.persona = persona
        self.input_wav = input_wav
        self.output_wav = output_wav
        self.output_json = output_json


@pytest.mark.asyncio
async def test_full_duplex_e2e_call(tmp_path):
    mock_port = 9911
    gw_port = 8788

    # 1. Start mock server
    mock = PersonaPlexMockServer(host="127.0.0.1", port=mock_port, prompt_init_delay=0.01, frame_interval_sec=0.02)
    await mock.start()

    try:
        # 2. Start Gateway
        pool = WorkerPool()
        pool.register_worker(WorkerNodeConfig(id="e2e-worker", host="127.0.0.1", port=mock_port))
        app = create_app(pool=pool)

        config = uvicorn.Config(app, host="127.0.0.1", port=gw_port, log_level="warning")
        server = uvicorn.Server(config)
        server_task = asyncio.create_task(server.serve())

        await asyncio.sleep(0.3)

        # 3. Create test input WAV
        input_wav_path = str(tmp_path / "test_user_in.wav")
        output_wav_path = str(tmp_path / "test_agent_out.wav")
        output_json_path = str(tmp_path / "test_transcript.json")

        t = np.linspace(0, 0.5, int(0.5 * SAMPLE_RATE), endpoint=False)
        sine_wave = (0.2 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)
        sf.write(input_wav_path, sine_wave, SAMPLE_RATE)

        # 4. Execute test call CLI routine
        args = MockArgs(
            url=f"ws://127.0.0.1:{gw_port}/v1/realtime",
            persona="citysan_service",
            input_wav=input_wav_path,
            output_wav=output_wav_path,
            output_json=output_json_path,
        )

        await _test_call_cmd(args)

        # 5. Verify outputs
        assert os.path.exists(output_wav_path)
        out_data, out_sr = sf.read(output_wav_path)
        assert out_sr == SAMPLE_RATE
        assert len(out_data) > 0

        assert os.path.exists(output_json_path)
        import json
        with open(output_json_path, "r", encoding="utf-8") as f:
            transcript_data = json.load(f)
            assert "transcript" in transcript_data
            assert len(transcript_data["tokens"]) > 0

        # Shutdown gateway
        server.should_exit = True
        await server_task

    finally:
        await mock.stop()
