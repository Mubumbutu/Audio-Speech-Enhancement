# lavasr_backend.py
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


def _chunk_fixed(audio_data: np.ndarray, sr: int, chunk_sec: float) -> list:
    chunk_samples = max(sr, int(chunk_sec * sr))
    total = len(audio_data)
    chunks = []
    pos = 0
    while pos < total:
        end = min(pos + chunk_samples, total)
        chunks.append(audio_data[pos:end].copy())
        pos = end
    return chunks


def _find_cut_point(audio_data: np.ndarray, sr: int, search_start: int, search_end: int) -> int:
    segment = audio_data[search_start:search_end]
    n = len(segment)
    if n == 0:
        return search_start
    window_samples = max(1, int(0.05 * sr))
    step = max(1, window_samples // 2)
    min_rms = float('inf')
    best_mid = n // 2
    i = 0
    while i + window_samples <= n:
        w = segment[i:i + window_samples]
        rms = float(np.dot(w, w) / window_samples) ** 0.5
        if rms < min_rms:
            min_rms = rms
            best_mid = i + window_samples // 2
        i += step
    return search_start + best_mid


def _chunk_smart(audio_data: np.ndarray, sr: int, chunk_sec: float) -> list:
    chunk_samples = max(sr, int(chunk_sec * sr))
    half_chunk = max(1, chunk_samples // 2)
    total = len(audio_data)
    chunks = []
    pos = 0
    while pos < total:
        ideal_end = pos + chunk_samples
        if ideal_end >= total:
            chunks.append(audio_data[pos:total].copy())
            break
        search_start = pos + half_chunk
        search_end = ideal_end
        if search_start >= search_end:
            cut = ideal_end
        else:
            cut = _find_cut_point(audio_data, sr, search_start, search_end)
        cut = max(pos + 1, min(cut, total))
        chunks.append(audio_data[pos:cut].copy())
        pos = cut
    return chunks


def _crossfade_join(a: np.ndarray, b: np.ndarray, fade_samples: int) -> np.ndarray:
    fs = min(fade_samples, len(a) // 2, len(b) // 2)
    if fs <= 0:
        return np.concatenate([a, b])
    fade_out = np.linspace(1.0, 0.0, fs, dtype=np.float32)
    fade_in = np.linspace(0.0, 1.0, fs, dtype=np.float32)
    overlap = a[-fs:] * fade_out + b[:fs] * fade_in
    return np.concatenate([a[:-fs], overlap, b[fs:]])


def _crossfade_concat(chunks: list, sr: int, fade_ms: int = 30) -> np.ndarray:
    if not chunks:
        return np.zeros(0, dtype=np.float32)
    if len(chunks) == 1:
        return chunks[0].astype(np.float32)
    fade_samples = int(fade_ms * sr / 1000)
    result = chunks[0].astype(np.float32)
    for chunk in chunks[1:]:
        result = _crossfade_join(result, chunk.astype(np.float32), fade_samples)
    return result


@register_backend
class LavaSRBackend(AudioEnhancerBackend):
    @property
    def name(self) -> str:
        return "LavaSR"

    @property
    def model_id(self) -> str:
        return "lavasr"

    @property
    def display_name(self) -> str:
        return "LavaSR Super-Resolution"

    @property
    def venv_names(self) -> List[str]:
        return ["venv_lavasr"]

    @property
    def download_repo(self) -> str:
        return "YatharthS/LavaSR"

    @property
    def download_size(self) -> str:
        return "~50 MB"

    @property
    def header_icon(self) -> str:
        return "🌋"

    @property
    def header_title(self) -> str:
        return "LavaSR Super-Resolution"

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
        return True

    @property
    def processing_params(self) -> List[dict]:
        return [
            {
                "key": "input_sr",
                "type": "choice",
                "label": "Input Sample Rate",
                "options": [
                    (8000,  "8 kHz"),
                    (16000, "16 kHz"),
                    (24000, "24 kHz"),
                    (44100, "44.1 kHz"),
                    (48000, "48 kHz"),
                ],
                "default": 16000,
                "tooltip": (
                    "The effective sample rate of the input audio content.\n"
                    "LavaSR supports any input SR from 8 kHz to 48 kHz.\n"
                    "Set this to match the actual bandwidth of your source,\n"
                    "not the file's sample rate. For example, a 44.1 kHz file\n"
                    "with content only up to 8 kHz should use 16 kHz here.\n"
                    "Output is always 48 kHz."
                ),
            },
            {
                "key": "denoise",
                "type": "bool",
                "label": "Denoise (remove background noise)",
                "default": False,
                "tooltip": (
                    "Enable noise suppression before upsampling.\n"
                    "Use only if your audio has background noise.\n"
                    "Disable for clean recordings to preserve natural sound."
                ),
            },
            {
                "key": "chunk_mode",
                "type": "choice",
                "label": "Chunking",
                "options": [
                    ("none",  "None (process whole file)"),
                    ("smart", "Smart (split at silence)"),
                    ("fixed", "Fixed (equal-length chunks)"),
                ],
                "default": "none",
                "tooltip": (
                    "Split long audio into chunks before processing.\n"
                    "None: process the whole file at once (fastest for short clips).\n"
                    "Smart: finds quiet points between chunks to minimise artefacts.\n"
                    "Fixed: splits into equal-length chunks.\n"
                    "Recommended: Smart for music/speech, Fixed for uniform content."
                ),
            },
            {
                "key": "chunk_sec",
                "type": "double_spin",
                "label": "Length",
                "min": 5.0,
                "max": 120.0,
                "step": 5.0,
                "decimals": 1,
                "suffix": " s",
                "default": 30.0,
                "tooltip": (
                    "Duration of each audio chunk in seconds.\n"
                    "Shorter chunks use less memory but produce more join points.\n"
                    "Longer chunks sound more natural.\n"
                    "Recommended: 30 s. Default: 30 s."
                ),
            },
        ]

    def is_available(self) -> bool:
        try:
            from LavaSR.model import LavaEnhance2  # noqa: F401
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
                "Run install/lavasr_install.bat or: pip install huggingface_hub")

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
            progress_cb("✅ LavaSR model downloaded successfully.")

    def _patch_vocos(self) -> None:
        try:
            import inspect
            import torchaudio
            from vocos.feature_extractors import MelSpectrogramFeatures
            if "f_min" in inspect.signature(MelSpectrogramFeatures.__init__).parameters:
                return
            def _new_init(inst, sample_rate, n_fft, hop_length, n_mels,
                          padding="center", f_min=0.0, f_max=None,
                          norm="slaney", mel_scale="slaney", **kwargs):
                super(MelSpectrogramFeatures, inst).__init__()
                if padding not in ["center", "same"]:
                    raise ValueError("Padding must be 'center' or 'same'.")
                inst.mel_spec = torchaudio.transforms.MelSpectrogram(
                    sample_rate,
                    n_fft=n_fft,
                    hop_length=hop_length,
                    n_mels=n_mels,
                    center=(padding == "center"),
                    power=1,
                    norm=norm,
                    mel_scale=mel_scale,
                    f_min=f_min,
                    f_max=f_max,
                )
                inst.padding = padding
                inst.n_fft = n_fft
            MelSpectrogramFeatures.__init__ = _new_init
        except Exception:
            pass

    def _load_model(self, model_dir: Path, device: str,
                    progress_cb: Optional[Callable[[str], None]] = None):
        try:
            from LavaSR.model import LavaEnhance2
        except ImportError:
            raise RuntimeError(
                "LavaSR is not installed.\n"
                "Run install/lavasr_install.bat or:\n"
                "  pip install git+https://github.com/ysharma3501/LavaSR.git")

        self._patch_vocos()

        if progress_cb:
            progress_cb(f"LavaSR: loading model on {device.upper()}…")

        return LavaEnhance2(str(model_dir), device)

    def _enhance_chunk(self, model, chunk: np.ndarray, chunk_sr: int,
                       input_sr: int, denoise: bool) -> np.ndarray:
        tmp_path = None
        try:
            import tempfile as _tmp
            fd, tmp_path = _tmp.mkstemp(suffix=".wav", prefix="lavasr_chunk_")
            import os
            os.close(fd)
            sf.write(tmp_path, chunk, chunk_sr)
            audio_tensor, _ = model.load_audio(tmp_path, input_sr=input_sr)
            out = model.enhance(audio_tensor, denoise=denoise)
            return out.cpu().numpy().squeeze().astype(np.float32)
        finally:
            if tmp_path:
                try:
                    Path(tmp_path).unlink()
                except Exception:
                    pass

    def process(self, request: EnhancementRequest,
                progress_cb: Optional[Callable[[str], None]] = None,
                log_cb: Optional[Callable[[str], None]] = None) -> EnhancementResult:
        import torch

        model_dir  = Path(request.params.get("model_dir", ""))
        input_sr   = int(request.params.get("input_sr", 16000))
        denoise    = bool(request.params.get("denoise", False))
        chunk_mode = str(request.params.get("chunk_mode", "none"))
        chunk_sec  = float(request.params.get("chunk_sec", 30.0))
        input_path = request.input_path
        output_dir = Path(request.output_dir)
        stem       = Path(input_path).stem

        if not model_dir.is_dir() or not any(model_dir.iterdir()):
            raise RuntimeError(
                "Model directory not found or empty.\n"
                "Click 'Download Model' to fetch LavaSR weights first.")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model  = self._load_model(model_dir, device, progress_cb)

        out_path = str(output_dir / f"lavasr_{stem}.wav")

        if chunk_mode == "none":
            if progress_cb:
                progress_cb(
                    f"LavaSR: running super-resolution "
                    f"(input SR={input_sr} Hz"
                    f"{', denoise=on' if denoise else ''})…")
            audio_tensor, _ = model.load_audio(input_path, input_sr=input_sr)
            output_tensor   = model.enhance(audio_tensor, denoise=denoise)
            output_audio    = output_tensor.cpu().numpy().squeeze().astype(np.float32)
            sf.write(out_path, output_audio, 48000)
            duration = len(output_audio) / 48000
        else:
            audio, sr = sf.read(input_path, dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)

            if chunk_mode == "fixed":
                chunks = _chunk_fixed(audio, sr, chunk_sec)
            else:
                chunks = _chunk_smart(audio, sr, chunk_sec)

            enhanced_chunks = []
            for i, chunk in enumerate(chunks):
                if progress_cb:
                    progress_cb(f"LavaSR: chunk {i + 1}/{len(chunks)}…")
                enhanced_chunks.append(
                    self._enhance_chunk(model, chunk, sr, input_sr, denoise))

            if progress_cb:
                progress_cb("LavaSR: assembling chunks with crossfade…")
            joined   = _crossfade_concat(enhanced_chunks, 48000, fade_ms=30)
            sf.write(out_path, joined, 48000)
            duration = len(joined) / 48000

        if progress_cb:
            progress_cb(f"✅ LavaSR complete: lavasr_{stem}.wav")

        return EnhancementResult(output_path=out_path, sample_rate=48000, duration_s=duration)
