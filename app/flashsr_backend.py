# flashsr.py
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
class FlashSRBackend(AudioEnhancerBackend):
    @property
    def name(self) -> str:
        return "FlashSR"

    @property
    def model_id(self) -> str:
        return "flashsr"

    @property
    def display_name(self) -> str:
        return "FlashSR Super-Resolution"

    @property
    def venv_names(self) -> List[str]:
        return ["venv_flashsr"]

    @property
    def download_repo(self) -> str:
        return "YatharthS/FlashSR"

    @property
    def download_size(self) -> str:
        return "~60 MB"

    @property
    def header_icon(self) -> str:
        return "⚡"

    @property
    def header_title(self) -> str:
        return "FlashSR Super-Resolution"

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
                "key": "backend_mode",
                "type": "choice",
                "label": "Backend",
                "options": [
                    ("onnx", "ONNX (CPU/GPU, fast, ~500 KB)"),
                    ("pytorch", "PyTorch (GPU recommended, ~60 MB)"),
                ],
                "default": "onnx",
                "tooltip": (
                    "ONNX: uses onnxruntime for inference. Extremely fast (~200-400x realtime).\n"
                    "     Tiny model (~500 KB). Works well on CPU.\n"
                    "PyTorch: uses the full FASR model (~60 MB). Requires GPU for best speed.\n"
                    "     Runs model weights in float16 (half precision)."
                ),
            },
        ]

    def _onnx_model_path(self, model_dir: Path) -> Path:
        return model_dir / "onnx" / "model.onnx"

    def _pytorch_model_path(self, model_dir: Path) -> Path:
        return model_dir / "upsampler.pth"

    def is_available(self) -> bool:
        try:
            import librosa  # noqa: F401
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
            from huggingface_hub import hf_hub_download
        except ImportError:
            raise RuntimeError(
                "huggingface_hub is not installed.\n"
                "Run install/flashsr_install.bat to set up the environment.")

        model_dir.mkdir(parents=True, exist_ok=True)
        onnx_dir = model_dir / "onnx"
        onnx_dir.mkdir(parents=True, exist_ok=True)

        onnx_dest = self._onnx_model_path(model_dir)
        pth_dest = self._pytorch_model_path(model_dir)

        if not onnx_dest.exists():
            if progress_cb:
                progress_cb("Downloading FlashSR ONNX model (~500 KB)…")
            hf_hub_download(
                repo_id="YatharthS/FlashSR",
                filename="model.onnx",
                subfolder="onnx",
                local_dir=str(model_dir),
            )
            if progress_cb:
                progress_cb("✅ ONNX model downloaded.")
        else:
            if progress_cb:
                progress_cb("ONNX model already present, skipping.")

        if not pth_dest.exists():
            if progress_cb:
                progress_cb("Downloading FlashSR PyTorch weights (~60 MB)…")
            hf_hub_download(
                repo_id="YatharthS/FlashSR",
                filename="upsampler.pth",
                local_dir=str(model_dir),
            )
            if progress_cb:
                progress_cb("✅ PyTorch weights downloaded.")
        else:
            if progress_cb:
                progress_cb("PyTorch weights already present, skipping.")

        if progress_cb:
            progress_cb("✅ FlashSR models ready.")

    def _model_ready(self, model_dir: Path, mode: str) -> bool:
        if mode == "onnx":
            return self._onnx_model_path(model_dir).exists()
        return self._pytorch_model_path(model_dir).exists()

    def _process_onnx(self, audio_16k: np.ndarray, model_dir: Path,
                      progress_cb: Optional[Callable[[str], None]] = None) -> np.ndarray:
        try:
            import onnxruntime as ort
        except ImportError:
            raise RuntimeError(
                "onnxruntime is not installed.\n"
                "Run install/flashsr_install.bat or: pip install onnxruntime")

        onnx_path = self._onnx_model_path(model_dir)
        if not onnx_path.exists():
            raise RuntimeError(
                f"ONNX model not found at: {onnx_path}\n"
                "Click 'Download Model' to fetch it first.")

        if progress_cb:
            progress_cb("FlashSR: running ONNX inference…")

        session = ort.InferenceSession(str(onnx_path))
        lowres = audio_16k[np.newaxis, :].astype(np.float32)
        output = session.run(["reconstruction"], {"audio_values": lowres})[0]
        return output.squeeze(0)

    def _process_pytorch(self, audio_16k: np.ndarray, model_dir: Path,
                          progress_cb: Optional[Callable[[str], None]] = None) -> np.ndarray:
        try:
            from FastAudioSR import FASR
        except ImportError:
            raise RuntimeError(
                "FastAudioSR is not installed.\n"
                "Run install/flashsr_install.bat or:\n"
                "  pip install git+https://github.com/ysharma3501/FlashSR.git")

        import torch

        pth_path = self._pytorch_model_path(model_dir)
        if not pth_path.exists():
            raise RuntimeError(
                f"PyTorch weights not found at: {pth_path}\n"
                "Click 'Download Model' to fetch them first.")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        if progress_cb:
            progress_cb(f"FlashSR: loading PyTorch model on {device.upper()}…")

        upsampler = FASR(str(pth_path))
        _ = upsampler.model.half()

        if progress_cb:
            progress_cb("FlashSR: running PyTorch inference…")

        lowres_wav = torch.from_numpy(audio_16k).unsqueeze(0).half()
        if device == "cuda":
            lowres_wav = lowres_wav.cuda()
            upsampler.model = upsampler.model.cuda()

        with torch.no_grad():
            result = upsampler.run(lowres_wav)

        if isinstance(result, torch.Tensor):
            result = result.cpu().float().numpy()
        if result.ndim > 1:
            result = result.squeeze(0)
        return result.astype(np.float32)

    def process(self, request: EnhancementRequest,
                progress_cb: Optional[Callable[[str], None]] = None,
                log_cb: Optional[Callable[[str], None]] = None) -> EnhancementResult:
        try:
            import librosa
        except ImportError:
            raise RuntimeError(
                "librosa is not installed.\n"
                "Run install/flashsr_install.bat to set up the environment.")

        mode = str(request.params.get("backend_mode", "onnx"))
        model_dir = Path(request.params.get("model_dir", ""))
        input_path = request.input_path
        output_dir = Path(request.output_dir)
        stem = Path(input_path).stem

        if not model_dir.is_dir():
            raise RuntimeError(
                "Model directory not configured.\n"
                "Click 'Download Model' to fetch FlashSR weights first.")

        if not self._model_ready(model_dir, mode):
            raise RuntimeError(
                f"FlashSR {'ONNX' if mode == 'onnx' else 'PyTorch'} model not found.\n"
                "Click 'Download Model' to fetch it first.")

        if progress_cb:
            progress_cb("FlashSR: resampling input to 16 kHz…")

        audio_16k, _ = librosa.load(input_path, sr=16000, mono=True)

        if mode == "onnx":
            enhanced = self._process_onnx(audio_16k, model_dir, progress_cb)
        else:
            enhanced = self._process_pytorch(audio_16k, model_dir, progress_cb)

        out_path = str(output_dir / f"flashsr_{stem}.wav")
        sf.write(out_path, enhanced.astype(np.float32), 48000)
        duration = len(enhanced) / 48000

        if progress_cb:
            progress_cb(f"✅ FlashSR complete: flashsr_{stem}.wav")

        return EnhancementResult(output_path=out_path, sample_rate=48000, duration_s=duration)
