# voicefixer_backend.py
import os
import tempfile
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
class VoiceFixerBackend(AudioEnhancerBackend):
    @property
    def name(self) -> str:
        return "VoiceFixer"

    @property
    def model_id(self) -> str:
        return "voicefixer"

    @property
    def display_name(self) -> str:
        return "VoiceFixer Speech Restoration"

    @property
    def venv_names(self) -> List[str]:
        return ["venv_voicefixer"]

    @property
    def download_repo(self) -> str:
        return "haoheliu/voicefixer"

    @property
    def download_size(self) -> str:
        return "~600 MB"

    @property
    def header_icon(self) -> str:
        return "🔧"

    @property
    def header_title(self) -> str:
        return "VoiceFixer Speech Restoration"

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
                "key": "mode",
                "type": "choice",
                "label": "Restoration Mode",
                "options": [
                    (0, "Mode 0 – Original model (recommended)"),
                    (1, "Mode 1 – Preprocessing: remove high frequencies"),
                    (2, "Mode 2 – Train mode (seriously degraded speech)"),
                ],
                "default": 0,
                "tooltip": (
                    "Mode 0: Standard restoration. Best for most cases.\n"
                    "        Handles noise, reverb, clipping and low sample rate.\n"
                    "Mode 1: Adds a preprocessing step that removes high-frequency\n"
                    "        content before restoration. Use when input has artefacts\n"
                    "        or content above the intended bandwidth.\n"
                    "Mode 2: Train mode. May work better on seriously degraded\n"
                    "        real-world speech where other modes struggle."
                ),
            },
        ]

    def _analysis_ckpt(self, model_dir: Path) -> Path:
        return model_dir / "analysis_module" / "checkpoints" / "vf.ckpt"

    def _synthesis_ckpt(self, model_dir: Path) -> Path:
        return model_dir / "synthesis_module" / "44100" / "model.ckpt-1490000_trimed.pt"

    def is_available(self) -> bool:
        try:
            from voicefixer import VoiceFixer  # noqa: F401
        except ImportError:
            return False
        return True

    def load(self, progress_cb: Optional[Callable[[str], None]] = None) -> None:
        return None

    def unload(self) -> None:
        return None

    def download(self, model_dir: Path,
                 progress_cb: Optional[Callable[[str], None]] = None) -> None:
        try:
            from voicefixer import VoiceFixer
        except ImportError:
            raise RuntimeError(
                "voicefixer is not installed.\n"
                "Run install/voicefixer_install.bat to set up the environment.")

        model_dir.mkdir(parents=True, exist_ok=True)

        analysis_ckpt  = self._analysis_ckpt(model_dir)
        synthesis_ckpt = self._synthesis_ckpt(model_dir)

        if analysis_ckpt.exists() and synthesis_ckpt.exists():
            if progress_cb:
                progress_cb("VoiceFixer checkpoints already present, skipping download.")
            return

        if progress_cb:
            progress_cb(
                "Downloading VoiceFixer checkpoints (~600 MB total)…\n"
                "  vf.ckpt (analysis module)\n"
                "  model.ckpt-1490000_trimed.pt (44.1 kHz vocoder)\n"
                "Please wait — this may take several minutes.")

        cache_analysis  = Path.home() / ".cache" / "voicefixer" / "analysis_module" / "checkpoints"
        cache_synthesis = Path.home() / ".cache" / "voicefixer" / "synthesis_module" / "44100"

        vf = VoiceFixer()
        del vf

        analysis_src  = cache_analysis  / "vf.ckpt"
        synthesis_src = cache_synthesis / "model.ckpt-1490000_trimed.pt"

        if analysis_src.exists():
            analysis_ckpt.parent.mkdir(parents=True, exist_ok=True)
            import shutil
            shutil.copy2(str(analysis_src), str(analysis_ckpt))
            if progress_cb:
                progress_cb("✅ Analysis checkpoint copied to model folder.")
        else:
            if progress_cb:
                progress_cb(
                    "⚠ vf.ckpt not found in cache. "
                    "VoiceFixer will download it automatically on first use.")

        if synthesis_src.exists():
            synthesis_ckpt.parent.mkdir(parents=True, exist_ok=True)
            import shutil
            shutil.copy2(str(synthesis_src), str(synthesis_ckpt))
            if progress_cb:
                progress_cb("✅ Synthesis checkpoint copied to model folder.")
        else:
            if progress_cb:
                progress_cb(
                    "⚠ model.ckpt-1490000_trimed.pt not found in cache. "
                    "VoiceFixer will download it automatically on first use.")

        if progress_cb:
            progress_cb("✅ VoiceFixer model download complete.")

    def _model_ready(self, model_dir: Path) -> bool:
        return (
            self._analysis_ckpt(model_dir).exists()
            and self._synthesis_ckpt(model_dir).exists()
        )

    def _redirect_cache(self, model_dir: Path) -> None:
        cache_analysis  = Path.home() / ".cache" / "voicefixer" / "analysis_module" / "checkpoints"
        cache_synthesis = Path.home() / ".cache" / "voicefixer" / "synthesis_module" / "44100"

        analysis_src  = model_dir / "analysis_module" / "checkpoints" / "vf.ckpt"
        synthesis_src = model_dir / "synthesis_module" / "44100" / "model.ckpt-1490000_trimed.pt"

        import shutil

        if analysis_src.exists() and not (cache_analysis / "vf.ckpt").exists():
            cache_analysis.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(analysis_src), str(cache_analysis / "vf.ckpt"))

        if synthesis_src.exists() and not (cache_synthesis / "model.ckpt-1490000_trimed.pt").exists():
            cache_synthesis.mkdir(parents=True, exist_ok=True)
            shutil.copy2(
                str(synthesis_src),
                str(cache_synthesis / "model.ckpt-1490000_trimed.pt"))

    def process(self, request: EnhancementRequest,
                progress_cb: Optional[Callable[[str], None]] = None,
                log_cb: Optional[Callable[[str], None]] = None) -> EnhancementResult:
        try:
            import torch
            from voicefixer import VoiceFixer
        except ImportError:
            raise RuntimeError(
                "voicefixer is not installed.\n"
                "Run install/voicefixer_install.bat to set up the environment.")

        model_dir  = Path(request.params.get("model_dir", ""))
        mode       = int(request.params.get("mode", 0))
        input_path = request.input_path
        output_dir = Path(request.output_dir)
        stem       = Path(input_path).stem

        if not model_dir.is_dir():
            raise RuntimeError(
                "Model directory not configured.\n"
                "Click 'Download Model' to fetch VoiceFixer checkpoints first.")

        self._redirect_cache(model_dir)

        cuda = torch.cuda.is_available()
        device_str = "GPU (CUDA)" if cuda else "CPU"

        if progress_cb:
            progress_cb(f"VoiceFixer: loading model on {device_str}…")

        vf = VoiceFixer()

        out_path = str(output_dir / f"voicefixer_mode{mode}_{stem}.wav")

        if progress_cb:
            progress_cb(
                f"VoiceFixer: restoring speech "
                f"(mode={mode}, device={device_str})…")

        vf.restore(
            input=input_path,
            output=out_path,
            cuda=cuda,
            mode=mode,
        )

        if not Path(out_path).exists():
            raise RuntimeError(
                f"VoiceFixer did not produce an output file at:\n{out_path}")

        data, sr = sf.read(out_path, dtype="float32")
        duration  = len(data) / max(1, sr)

        if progress_cb:
            progress_cb(f"✅ VoiceFixer complete: voicefixer_mode{mode}_{stem}.wav")

        return EnhancementResult(output_path=out_path, sample_rate=sr, duration_s=duration)
