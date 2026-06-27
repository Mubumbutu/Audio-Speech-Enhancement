# universr.py
import os
import shutil
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


def chunk_fixed(audio_data: np.ndarray, sr: int, chunk_sec: float) -> list:
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


def chunk_smart(audio_data: np.ndarray, sr: int, chunk_sec: float) -> list:
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


def crossfade_concat(chunks: list, sr: int, fade_ms: int = 30) -> np.ndarray:
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
class UniverSRBackend(AudioEnhancerBackend):
    @property
    def name(self) -> str:
        return "UniverSR"

    @property
    def model_id(self) -> str:
        return "universr"

    @property
    def display_name(self) -> str:
        return "UniverSR Super-Resolution"

    @property
    def venv_names(self) -> List[str]:
        return ["venv_universr"]

    @property
    def download_repo(self) -> str:
        return "woongzip1/universr-audio"

    @property
    def download_size(self) -> str:
        return "~300 MB"

    @property
    def header_icon(self) -> str:
        return "🔊"

    @property
    def header_title(self) -> str:
        return "UniverSR Super-Resolution"

    @property
    def requires_gpu(self) -> bool:
        return True

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
                    (12000, "12 kHz"),
                    (16000, "16 kHz"),
                    (24000, "24 kHz"),
                ],
                "default": 8000,
                "tooltip": (
                    "The effective sample rate of the input audio content.\n"
                    "Set this to match the actual bandwidth of your source,\n"
                    "not the file's sample rate. For example, a 48 kHz file\n"
                    "with content only up to 4 kHz should use 8 kHz here.\n"
                    "All inputs are upsampled to 48 kHz output.\n"
                    "Best quality at 8 kHz (70% of training data)."
                ),
            },
            {
                "key": "model_variant",
                "type": "choice",
                "label": "Model Variant",
                "options": [
                    ("woongzip1/universr-audio",  "universr-audio (general, recommended)"),
                    ("woongzip1/universr-speech", "universr-speech (speech only)"),
                ],
                "default": "woongzip1/universr-audio",
                "tooltip": (
                    "universr-audio: General audio model trained on speech, music,\n"
                    "and sound effects. Recommended for most use cases.\n"
                    "universr-speech: Speech-only model. May give better results\n"
                    "for clean speech recordings."
                ),
            },
            {
                "key": "ode_method",
                "type": "choice",
                "label": "ODE Method",
                "options": [
                    ("midpoint", "Midpoint (recommended)"),
                    ("euler",    "Euler (fastest)"),
                    ("rk4",     "RK4 (best quality)"),
                ],
                "default": "midpoint",
                "tooltip": (
                    "ODE integration method for the flow matching model.\n"
                    "Midpoint: best speed/quality balance (recommended).\n"
                    "Euler: fastest but lower quality.\n"
                    "RK4: highest quality but slowest."
                ),
            },
            {
                "key": "ode_steps",
                "type": "spin",
                "label": "ODE Steps",
                "min": 1,
                "max": 50,
                "step": 1,
                "default": 4,
                "tooltip": (
                    "Number of ODE integration steps.\n"
                    "Higher values = better quality but slower processing.\n"
                    "Recommended: 4 (midpoint) or 6 (euler). Default: 4."
                ),
            },
            {
                "key": "guidance_scale",
                "type": "double_spin",
                "label": "Guidance Scale",
                "min": 0.0,
                "max": 5.0,
                "step": 0.1,
                "decimals": 1,
                "default": 1.5,
                "tooltip": (
                    "Classifier-free guidance (CFG) scale.\n"
                    "0.0 = disabled (most faithful to input).\n"
                    "Speech: 1.0–1.5  |  Music: 1.5–2.0  |  SFX: 1.5\n"
                    "Higher values produce richer high-frequency content\n"
                    "but deviate more from the ground-truth reference."
                ),
            },
            {
                "key": "chunk_mode",
                "type": "choice",
                "label": "Chunking",
                "options": [("none", "Off"), ("fixed", "Fixed"), ("smart", "Smart")],
                "default": "none",
                "tooltip": (
                    "Off: super-resolve the entire file in one pass.\n"
                    "Fixed: split audio into equal-length segments, then join with crossfade.\n"
                    "Smart: split at the quietest point in each window to avoid cutting audio.\n"
                    "Use chunking for very long files to manage GPU memory."
                ),
            },
            {
                "key": "chunk_sec",
                "type": "double_spin",
                "label": "Length",
                "min": 5.0,
                "max": 60.0,
                "step": 5.0,
                "decimals": 1,
                "suffix": " s",
                "default": 15.0,
                "tooltip": (
                    "Duration of each audio chunk in seconds.\n"
                    "Shorter chunks use less GPU memory but produce more join points.\n"
                    "Longer chunks sound more natural but may require more VRAM.\n"
                    "Recommended: 15–30 s. Default: 15 s."
                ),
            },
        ]

    def is_available(self) -> bool:
        try:
            from universr import UniverSR  # noqa: F401
            import torch  # noqa: F401
        except ImportError:
            return False
        return True

    def load(self, progress_cb: Optional[Callable[[str], None]] = None) -> None:
        return None

    def unload(self) -> None:
        return None

    def download(self, model_dir: Path,
                 progress_cb: Optional[Callable[[str], None]] = None) -> None:
        if progress_cb:
            progress_cb("Importing huggingface_hub…")
        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            raise RuntimeError(
                "huggingface_hub is not installed.\n"
                "Run install/universr_install.bat or: pip install huggingface_hub")
        repo_id = self.download_repo
        model_dir.mkdir(parents=True, exist_ok=True)
        if progress_cb:
            progress_cb(
                f"Downloading {repo_id} from HuggingFace…  "
                f"(≈ {self.download_size} — please wait)")
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(model_dir),
            local_dir_use_symlinks=False,
        )
        if progress_cb:
            progress_cb("✅ Model downloaded successfully.")

    def _load_model(self, model_dir: Path,
                    model_variant: str,
                    progress_cb: Optional[Callable[[str], None]] = None):
        try:
            from universr import UniverSR
        except ImportError:
            raise RuntimeError(
                "universr is not installed.\n"
                "Run install/universr_install.bat or:\n"
                "pip install git+https://github.com/woongzip1/UniverSR.git")

        import torch

        if not torch.cuda.is_available():
            raise RuntimeError(
                "No CUDA GPU detected.\n"
                "UniverSR requires an NVIDIA GPU with CUDA.\n"
                "CPU-only mode is not supported.")

        ckpt_exts = (".pth", ".pt", ".ckpt", ".bin")
        local_ckpt = None
        for ext in ckpt_exts:
            found = sorted(model_dir.rglob(f"*{ext}"))
            if found:
                local_ckpt = found[0]
                break

        config_path = None
        for name in ("config.yaml", "config.yml"):
            p = model_dir / name
            if p.exists():
                config_path = p
                break
        if config_path is None:
            found_cfgs = list(model_dir.rglob("config.yaml")) + list(model_dir.rglob("config.yml"))
            if found_cfgs:
                config_path = found_cfgs[0]

        if local_ckpt is not None and config_path is not None:
            if progress_cb:
                progress_cb(f"UniverSR: loading from local checkpoint {local_ckpt.name}…")
            model = UniverSR.from_local(
                ckpt_path=str(local_ckpt),
                config_path=str(config_path),
                device="cuda",
            )
        else:
            if progress_cb:
                progress_cb(f"UniverSR: loading {model_variant} from HuggingFace…")
            model = UniverSR.from_pretrained(model_variant, device="cuda")

        return model

    def process(self, request: EnhancementRequest,
                progress_cb: Optional[Callable[[str], None]] = None,
                log_cb: Optional[Callable[[str], None]] = None) -> EnhancementResult:
        model_dir = Path(request.params["model_dir"])
        input_sr = int(request.params.get("input_sr", 8000))
        model_variant = request.params.get("model_variant", "woongzip1/universr-audio")
        ode_method = request.params.get("ode_method", "midpoint")
        ode_steps = int(request.params.get("ode_steps", 4))
        guidance_scale_raw = request.params.get("guidance_scale", 1.5)
        guidance_scale = float(guidance_scale_raw) if float(guidance_scale_raw) > 0.0 else None
        chunk_mode = request.params.get("chunk_mode", "none")
        chunk_sec = float(request.params.get("chunk_sec", 15.0))
        output_dir = Path(request.output_dir)

        import torch
        import torchaudio

        if not torch.cuda.is_available():
            raise RuntimeError(
                "No CUDA GPU detected.\n"
                "UniverSR requires an NVIDIA GPU with CUDA.\n"
                "CPU-only mode is not supported.")

        if progress_cb:
            progress_cb("UniverSR: loading model on CUDA…")

        model = self._load_model(model_dir, model_variant, progress_cb)

        input_path = request.input_path
        stem = Path(input_path).stem
        out_path = str(output_dir / f"universr_{stem}.wav")

        src_ext = Path(input_path).suffix.lower()
        needs_convert = src_ext not in (".wav", ".flac")
        actual_input = input_path
        tmp_wav = None

        if needs_convert:
            if progress_cb:
                progress_cb("UniverSR: converting input to WAV…")
            import subprocess
            fd, tmp_wav = tempfile.mkstemp(suffix=".wav", prefix="universr_in_")
            os.close(fd)
            r = subprocess.run(
                ["ffmpeg", "-y", "-i", input_path, "-ac", "1", tmp_wav],
                capture_output=True, timeout=300,
            )
            if r.returncode != 0:
                raise RuntimeError(
                    f"ffmpeg conversion failed:\n"
                    f"{r.stderr.decode(errors='replace')[-400:]}")
            actual_input = tmp_wav

        try:
            if chunk_mode == "none":
                if progress_cb:
                    progress_cb(
                        f"UniverSR: running super-resolution "
                        f"({ode_method}, {ode_steps} steps, guidance={guidance_scale})…")
                output = model.enhance(
                    actual_input,
                    input_sr=input_sr,
                    ode_method=ode_method,
                    ode_steps=ode_steps,
                    guidance_scale=guidance_scale,
                )
                torchaudio.save(out_path, output.cpu(), 48000)
                data_len = output.shape[-1]
                duration = data_len / 48000
            else:
                audio, sr = sf.read(actual_input, dtype="float32")
                if audio.ndim > 1:
                    audio = audio.mean(axis=1)

                chunks = (chunk_fixed(audio, sr, chunk_sec)
                          if chunk_mode == "fixed"
                          else chunk_smart(audio, sr, chunk_sec))

                tmp_dir = Path(tempfile.mkdtemp(prefix="universr_chunks_"))
                try:
                    enhanced_chunks = []
                    for i, chunk in enumerate(chunks):
                        if progress_cb:
                            progress_cb(f"UniverSR: chunk {i + 1}/{len(chunks)}…")
                        chunk_path = tmp_dir / f"chunk_{i:04d}.wav"
                        sf.write(str(chunk_path), chunk, sr)
                        output = model.enhance(
                            str(chunk_path),
                            input_sr=input_sr,
                            ode_method=ode_method,
                            ode_steps=ode_steps,
                            guidance_scale=guidance_scale,
                        )
                        enh_np = output.squeeze(0).cpu().numpy()
                        enhanced_chunks.append(enh_np.astype(np.float32))
                finally:
                    shutil.rmtree(str(tmp_dir), ignore_errors=True)

                if progress_cb:
                    progress_cb("UniverSR: assembling chunks with crossfade…")
                joined = crossfade_concat(enhanced_chunks, 48000, fade_ms=30)
                sf.write(out_path, joined, 48000)
                duration = len(joined) / 48000
        finally:
            if tmp_wav and Path(tmp_wav).exists():
                try:
                    Path(tmp_wav).unlink()
                except Exception:
                    pass

        if progress_cb:
            progress_cb("✅ UniverSR complete.")
        return EnhancementResult(output_path=out_path, sample_rate=48000, duration_s=duration)
