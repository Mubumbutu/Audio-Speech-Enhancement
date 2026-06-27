# flowhigh_backend.py
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
class FlowHighBackend(AudioEnhancerBackend):
    @property
    def name(self) -> str:
        return "FlowHigh"

    @property
    def model_id(self) -> str:
        return "flowhigh"

    @property
    def display_name(self) -> str:
        return "FlowHigh Super-Resolution"

    @property
    def venv_names(self) -> List[str]:
        return ["venv_flowhigh"]

    @property
    def download_repo(self) -> str:
        return "ResembleAI/FlowHigh"

    @property
    def download_size(self) -> str:
        return "~500 MB"

    @property
    def header_icon(self) -> str:
        return "🌊"

    @property
    def header_title(self) -> str:
        return "FlowHigh Super-Resolution"

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
        return []

    def is_available(self) -> bool:
        try:
            from flowhigh import FlowHighSR  # noqa: F401
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
            from huggingface_hub import snapshot_download
        except ImportError:
            raise RuntimeError(
                "huggingface_hub is not installed.\n"
                "Run install/flowhigh_install.bat or: pip install huggingface_hub")

        model_dir.mkdir(parents=True, exist_ok=True)

        if progress_cb:
            progress_cb(
                f"Downloading {self.download_repo} from HuggingFace…  "
                f"(≈ {self.download_size} — please wait)")

        snapshot_download(
            repo_id=self.download_repo,
            local_dir=str(model_dir),
            local_dir_use_symlinks=False,
        )

        if progress_cb:
            progress_cb("✅ FlowHigh model downloaded successfully.")

    def process(self, request: EnhancementRequest,
                progress_cb: Optional[Callable[[str], None]] = None,
                log_cb: Optional[Callable[[str], None]] = None) -> EnhancementResult:
        try:
            import torch
            import torchaudio as ta
            from flowhigh import FlowHighSR
        except ImportError:
            raise RuntimeError(
                "flowhigh or torchaudio is not installed.\n"
                "Run install/flowhigh_install.bat to set up the environment.")

        model_dir = Path(request.params.get("model_dir", ""))
        input_path = request.input_path
        output_dir = Path(request.output_dir)
        stem = Path(input_path).stem

        if not model_dir.is_dir() or not any(model_dir.iterdir()):
            raise RuntimeError(
                "Model directory not found or empty.\n"
                "Click 'Download Model' to fetch FlowHigh weights first.")

        device = "cuda" if torch.cuda.is_available() else "cpu"

        if progress_cb:
            progress_cb(f"FlowHigh: loading model on {device.upper()}…")

        model = FlowHighSR.from_local(model_dir, device)

        if progress_cb:
            progress_cb("FlowHigh: loading audio…")

        wav, sr_in = ta.load(input_path)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0)
        else:
            wav = wav.squeeze(0)
        wav_np = wav.numpy()

        if progress_cb:
            progress_cb(f"FlowHigh: running super-resolution (input SR={sr_in} Hz)…")

        wav_hr = model.generate(wav_np, sr_in, 48000)

        out_path = str(output_dir / f"flowhigh_{stem}.wav")
        ta.save(out_path, wav_hr.cpu(), 48000)
        duration = wav_hr.shape[-1] / 48000

        if progress_cb:
            progress_cb(f"✅ FlowHigh complete: flowhigh_{stem}.wav")

        return EnhancementResult(output_path=out_path, sample_rate=48000, duration_s=duration)
