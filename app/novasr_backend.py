# novasr_backend.py
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np
import soundfile as sf

from models_backends import (
    AudioEnhancerBackend,
    EnhancementRequest,
    EnhancementResult,
    register_backend,
)


@register_backend
class NovaSRBackend(AudioEnhancerBackend):
    @property
    def name(self) -> str:
        return "NovaSR"

    @property
    def model_id(self) -> str:
        return "novasr"

    @property
    def display_name(self) -> str:
        return "NovaSR Super-Resolution"

    @property
    def venv_names(self) -> List[str]:
        return ["venv_novasr"]

    @property
    def download_repo(self) -> str:
        return ""

    @property
    def download_size(self) -> str:
        return ""

    @property
    def header_icon(self) -> str:
        return "⚡"

    @property
    def header_title(self) -> str:
        return "NovaSR Super-Resolution"

    @property
    def requires_gpu(self) -> bool:
        return False

    @property
    def auth_required(self) -> bool:
        return False

    @property
    def supports_demucs_preprocessing(self) -> bool:
        return False

    @property
    def supports_bandwidth_extension(self) -> bool:
        return False

    @property
    def supports_chunked_processing(self) -> bool:
        return False

    @property
    def processing_params(self) -> List[dict]:
        return [
            {
                "key": "use_half",
                "type": "bool",
                "label": "Half precision (GPU fp16 — faster, 2x less VRAM)",
                "default": True,
                "tooltip": (
                    "Enable fp16 (half-precision) inference.\n"
                    "Recommended for NVIDIA GPUs — roughly 2x faster with less VRAM.\n"
                    "Disable for CPU inference: NovaSR runs 3-4x faster in full precision on CPU.\n"
                    "Also disable if you see NaN/silence artifacts on older GPUs."
                ),
            },
        ]

    def is_available(self) -> bool:
        try:
            from NovaSR import FastSR  # noqa: F401
        except ImportError:
            return False
        return True

    def load(self, progress_cb: Optional[Callable[[str], None]] = None) -> None:
        return None

    def unload(self) -> None:
        return None

    def download(self, model_dir: Path,
                 progress_cb: Optional[Callable[[str], None]] = None) -> None:
        return None

    def process(self, request: EnhancementRequest,
                progress_cb: Optional[Callable[[str], None]] = None,
                log_cb: Optional[Callable[[str], None]] = None) -> EnhancementResult:
        try:
            from NovaSR import FastSR
        except ImportError as _ie:
            raise RuntimeError(
                "NovaSR is not installed.\n"
                "Run install/novasr_install.bat or:\n"
                "pip install git+https://github.com/ysharma3501/NovaSR.git"
            ) from _ie

        import torch

        use_half = bool(request.params.get("use_half", True))
        input_path = request.input_path
        output_dir = Path(request.output_dir)
        model_dir = Path(request.params.get("model_dir", str(output_dir)))

        on_gpu = torch.cuda.is_available()

        if use_half and not on_gpu:
            if progress_cb:
                progress_cb(
                    "NovaSR: half precision disabled (no CUDA GPU detected — using CPU fp32)."
                )
            use_half = False

        cache_dir = model_dir / "novasr_hf_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)

        if progress_cb:
            progress_cb(
                "NovaSR: loading model"
                + (" (fp16, GPU)" if use_half else " (fp32, CPU)" if not on_gpu else " (fp32, GPU)")
                + "…  (downloads ~52 KB from HuggingFace on first run)"
            )

        try:
            import os
            os.environ.setdefault("HF_HOME", str(cache_dir))
            upsampler = FastSR(half=use_half)
        except Exception as exc:
            raise RuntimeError(
                f"NovaSR failed to load: {exc}\n\n"
                "Re-run install/novasr_install.bat to repair the installation."
            ) from exc

        if progress_cb:
            progress_cb("NovaSR: loading audio…")

        try:
            lowres_audio = upsampler.load_audio(input_path)
        except Exception as exc:
            raise RuntimeError(
                f"NovaSR failed to load audio file: {exc}\n"
                "Make sure the file is a valid WAV or MP3.\n"
                "For other formats, ffmpeg must be installed and in PATH."
            ) from exc

        if progress_cb:
            progress_cb("NovaSR: running super-resolution (16 kHz → 48 kHz)…")

        highres_audio = upsampler.infer(lowres_audio).cpu()

        stem = Path(input_path).stem
        out_path = str(output_dir / f"novasr_{stem}.wav")

        audio_np = highres_audio.squeeze().numpy()
        if audio_np.ndim == 0:
            audio_np = audio_np.reshape(1)
        audio_np = audio_np.astype(np.float32)

        sf.write(out_path, audio_np, 48000)
        duration = len(audio_np) / 48000

        if progress_cb:
            progress_cb("✅ NovaSR complete.")

        return EnhancementResult(output_path=out_path, sample_rate=48000, duration_s=duration)
