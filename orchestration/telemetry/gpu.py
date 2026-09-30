"""
Live GPU telemetry and hardware metrics provider for PersonaPlex.
Extracts GPU utilization, VRAM usage, temperature, and stage devices.
"""

from __future__ import annotations
import subprocess
from typing import Dict, Any

_pynvml_initialized = False
_pynvml_handle = None

def _init_pynvml():
    global _pynvml_initialized, _pynvml_handle
    if _pynvml_initialized:
        return _pynvml_handle
    try:
        import pynvml
        pynvml.nvmlInit()
        _pynvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        _pynvml_initialized = True
        return _pynvml_handle
    except Exception:
        _pynvml_initialized = False
        return None


def get_live_gpu_telemetry() -> Dict[str, Any]:
    """Retrieve real-time GPU statistics via pynvml, torch or nvidia-smi."""
    info = {
        "cuda_available": False,
        "device_name": "None",
        "vram_used_mb": 0.0,
        "vram_total_mb": 0.0,
        "vram_free_mb": 0.0,
        "gpu_util_pct": 0,
        "temperature_c": None,
        "stages": {
            "vad": "cuda (Silero/Spectral)",
            "stt": "cuda (faster-whisper float16)",
            "llm": "cuda:0 (Qwen2.5 Ollama 100% GPU)",
            "chunker": "cpu (streaming parser)",
            "tts": "cuda / neural (float32 24kHz)",
        }
    }

    handle = _init_pynvml()
    if handle is not None:
        try:
            import pynvml
            info["cuda_available"] = True
            name = pynvml.nvmlDeviceGetName(handle)
            info["device_name"] = name if isinstance(name, str) else name.decode("utf-8")
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            info["vram_total_mb"] = round(mem.total / (1024 * 1024), 1)
            info["vram_used_mb"] = round(mem.used / (1024 * 1024), 1)
            info["vram_free_mb"] = round(mem.free / (1024 * 1024), 1)
            info["temperature_c"] = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            info["gpu_util_pct"] = util.gpu
            return info
        except Exception:
            pass

    # Fallback to torch.cuda
    try:
        import torch
        if torch.cuda.is_available():
            info["cuda_available"] = True
            info["device_name"] = torch.cuda.get_device_name(0)
            total = torch.cuda.get_device_properties(0).total_memory
            total_mb = round(total / (1024 * 1024), 1)
            used_mb = round(torch.cuda.memory_allocated(0) / (1024 * 1024), 1)
            info["vram_total_mb"] = total_mb
            info["vram_used_mb"] = used_mb
            info["vram_free_mb"] = round(total_mb - used_mb, 1)
            return info
    except Exception:
        pass

    # Fallback to nvidia-smi
    try:
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.used,memory.free,temperature.gpu,utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=3,
        )
        if res.returncode == 0:
            parts = [p.strip() for p in res.stdout.strip().split(",")]
            if len(parts) >= 4:
                info["cuda_available"] = True
                info["device_name"] = parts[0]
                info["vram_total_mb"] = float(parts[1])
                info["vram_used_mb"] = float(parts[2])
                info["vram_free_mb"] = float(parts[3])
                if len(parts) >= 5:
                    info["temperature_c"] = float(parts[4])
                if len(parts) >= 6:
                    info["gpu_util_pct"] = float(parts[5])
    except Exception:
        pass

    return info
