# audiosr.py
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
class AudioSRBackend(AudioEnhancerBackend):
    @property
    def name(self) -> str:
        return "AudioSR"

    @property
    def model_id(self) -> str:
        return "audiosr"

    @property
    def display_name(self) -> str:
        return "AudioSR Super-Resolution"

    @property
    def venv_names(self) -> List[str]:
        return ["venv_audiosr"]

    @property
    def download_repo(self) -> str:
        return ""

    @property
    def download_size(self) -> str:
        return ""

    @property
    def header_icon(self) -> str:
        return "🎚"

    @property
    def header_title(self) -> str:
        return "AudioSR Super-Resolution"

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
                "key": "chunk_mode",
                "type": "choice",
                "label": "Chunking",
                "options": [("none", "Off"), ("fixed", "Fixed"), ("smart", "Smart")],
                "default": "none",
                "tooltip": (
                    "Off: super-resolve the entire file in one pass.\n"
                    "Fixed: split audio into equal-length segments, then join with crossfade.\n"
                    "Smart: split at the quietest point in each window to avoid cutting speech.\n"
                    "AudioSR processes in 10-second windows internally; chunking is useful\n"
                    "for very long files to manage GPU memory and track progress."
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
            import audiosr  # noqa: F401
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
        return None

    def process(self, request: EnhancementRequest,
                progress_cb: Optional[Callable[[str], None]] = None,
                log_cb: Optional[Callable[[str], None]] = None) -> EnhancementResult:
        if progress_cb:
            progress_cb("Running AudioSR super-resolution…  (this may take several minutes)")
        try:
            from audiosr import super_resolution, build_model
        except ImportError as _ie:
            msg = str(_ie)
            if "audiosr" in msg or "No module named 'audiosr'" in msg:
                raise RuntimeError(
                    "audiosr is not installed.\n"
                    "Run install/audiosr_install.bat or: pip install audiosr==0.0.7 --no-deps")
            raise RuntimeError(
                f"audiosr failed to import due to a missing dependency:\n{msg}\n\n"
                "Re-run install/audiosr_install.bat to install all required audiosr dependencies.")

        import torch

        if not torch.cuda.is_available():
            raise RuntimeError(
                "No CUDA GPU detected.\n"
                "AudioSR requires an NVIDIA GPU with CUDA.\n"
                "CPU-only mode is not supported.")

        input_path = request.input_path
        chunk_mode = request.params.get("chunk_mode", "none")
        chunk_sec = float(request.params.get("chunk_sec", 15.0))
        output_dir = Path(request.output_dir)

        src_ext = Path(input_path).suffix.lower()
        needs_lowpass = src_ext in (".mp3", ".aac", ".ogg", ".opus", ".wma", ".m4a")
        actual_input = input_path

        if needs_lowpass:
            if progress_cb:
                progress_cb("AudioSR: applying low-pass filter for compressed input…")
            try:
                import torchaudio
                import torchaudio.functional as AF
                raw, raw_sr = sf.read(input_path, dtype="float32")
                if raw.ndim > 1:
                    raw = raw.mean(axis=1)
                t = torch.from_numpy(raw).unsqueeze(0)
                cutoff_hz = raw_sr * 0.45
                t_lp = AF.lowpass_biquad(t, sample_rate=raw_sr,
                                          cutoff_freq=cutoff_hz, Q=0.707)
                lp_tmp_fd, lp_tmp = tempfile.mkstemp(suffix=".wav", prefix="reuse_asr_lp_")
                os.close(lp_tmp_fd)
                sf.write(lp_tmp, t_lp.squeeze(0).numpy(), raw_sr)
                actual_input = lp_tmp
            except Exception:
                actual_input = input_path

        if progress_cb:
            progress_cb("AudioSR: loading model on CUDA…")
        model = build_model(model_name="basic", device="cuda")
        stem = Path(input_path).stem
        out_path = str(output_dir / f"audiosr_{stem}.wav")

        try:
            if chunk_mode == "none":
                if progress_cb:
                    progress_cb("AudioSR: running super-resolution (50 DDIM steps)…")
                waveform = super_resolution(model, actual_input, seed=42, ddim_steps=50)
                data = np.array(waveform)
                if data.ndim == 3:
                    data = data[0][0]
                elif data.ndim == 2:
                    data = data[0]
                sf.write(out_path, data, 48000)
                duration = len(data) / 48000
            else:
                audio, sr = sf.read(actual_input, dtype="float32")
                if audio.ndim > 1:
                    audio = audio.mean(axis=1)
                chunks = (chunk_fixed(audio, sr, chunk_sec)
                          if chunk_mode == "fixed"
                          else chunk_smart(audio, sr, chunk_sec))
                tmp_dir = Path(tempfile.mkdtemp(prefix="reuse_asr_"))
                try:
                    enhanced_chunks = []
                    for i, chunk in enumerate(chunks):
                        if progress_cb:
                            progress_cb(f"AudioSR: chunk {i + 1}/{len(chunks)}…")
                        chunk_path = tmp_dir / f"chunk_{i:04d}.wav"
                        sf.write(str(chunk_path), chunk, sr)
                        waveform = super_resolution(model, str(chunk_path), seed=42, ddim_steps=50)
                        data = np.array(waveform)
                        if data.ndim == 3:
                            data = data[0][0]
                        elif data.ndim == 2:
                            data = data[0]
                        enhanced_chunks.append(data.astype(np.float32))
                finally:
                    shutil.rmtree(str(tmp_dir), ignore_errors=True)
                if progress_cb:
                    progress_cb("AudioSR: assembling chunks with crossfade…")
                joined = crossfade_concat(enhanced_chunks, 48000, fade_ms=30)
                sf.write(out_path, joined, 48000)
                duration = len(joined) / 48000
        finally:
            if needs_lowpass and actual_input != input_path:
                try:
                    Path(actual_input).unlink()
                except Exception:
                    pass

        if progress_cb:
            progress_cb("✅ AudioSR complete.")
        return EnhancementResult(output_path=out_path, sample_rate=48000, duration_s=duration)
