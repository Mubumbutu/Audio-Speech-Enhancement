# deepfilternet.py
import os
import shutil
import tempfile
import urllib.request
import zipfile
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

_MODEL_URLS = {
    "DeepFilterNet2": "https://github.com/Rikorose/DeepFilterNet/raw/main/models/DeepFilterNet2.zip",
    "DeepFilterNet3": "https://github.com/Rikorose/DeepFilterNet/raw/main/models/DeepFilterNet3.zip",
}


def _download_model_zip(model_name: str, model_dir: Path,
                         progress_cb: Optional[Callable[[str], None]] = None) -> None:
    url = _MODEL_URLS[model_name]
    dest_dir = model_dir.parent
    dest_dir.mkdir(parents=True, exist_ok=True)
    zip_path = dest_dir / f"{model_name}.zip"

    if progress_cb:
        progress_cb(f"Downloading {model_name} from GitHub…")

    def _reporthook(block_num, block_size, total_size):
        if total_size > 0 and progress_cb:
            pct = min(100, int(block_num * block_size * 100 / total_size))
            progress_cb(f"Downloading {model_name}… {pct}%")

    urllib.request.urlretrieve(url, str(zip_path), reporthook=_reporthook)

    if progress_cb:
        progress_cb(f"Extracting {model_name}…")

    with zipfile.ZipFile(str(zip_path), "r") as zf:
        zf.extractall(str(dest_dir))

    try:
        zip_path.unlink()
    except Exception:
        pass

    if not model_dir.exists():
        for candidate in dest_dir.iterdir():
            if candidate.is_dir() and candidate.name.lower() == model_name.lower():
                candidate.rename(model_dir)
                break


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
class DeepFilterNetBackend(AudioEnhancerBackend):
    @property
    def name(self) -> str:
        return "DeepFilterNet"

    @property
    def model_id(self) -> str:
        return "deepfilternet"

    @property
    def display_name(self) -> str:
        return "DeepFilterNet Noise Suppression"

    @property
    def venv_names(self) -> List[str]:
        return ["venv_deepfilternet"]

    @property
    def download_repo(self) -> str:
        return "DeepFilterNet2 / DeepFilterNet3"

    @property
    def download_size(self) -> str:
        return "~2 MB"

    @property
    def header_icon(self) -> str:
        return "🔇"

    @property
    def header_title(self) -> str:
        return "DeepFilterNet Noise Suppression"

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
                "key": "df_model",
                "type": "choice",
                "label": "Model",
                "options": [
                    ("DeepFilterNet3", "DeepFilterNet3 (recommended)"),
                    ("DeepFilterNet2", "DeepFilterNet2 (stable)"),
                ],
                "default": "DeepFilterNet3",
                "tooltip": (
                    "DeepFilterNet3: latest model, best noise suppression quality.\n"
                    "DeepFilterNet2: previous generation, slightly lighter, very stable.\n"
                    "Both models operate at 48 kHz. Input is automatically resampled."
                ),
            },
            {
                "key": "atten_lim_db",
                "type": "spin",
                "label": "Attenuation limit",
                "min": 0,
                "max": 100,
                "step": 1,
                "default": 0,
                "tooltip": (
                    "Maximum noise attenuation in dB.\n"
                    "0 = no limit, apply full suppression (default).\n"
                    "e.g. 12 = suppress noise by at most 12 dB, keeping some residual noise.\n"
                    "Useful when full suppression sounds unnatural or over-processed."
                ),
            },
            {
                "key": "post_filter",
                "type": "bool",
                "label": "Post-filter",
                "default": False,
                "tooltip": (
                    "Enable an additional post-filter for extra noise reduction.\n"
                    "May slightly over-suppress quiet signals. Disabled by default."
                ),
            },
            {
                "key": "chunk_mode",
                "type": "choice",
                "label": "Chunking",
                "options": [("none", "Off"), ("fixed", "Fixed"), ("smart", "Smart")],
                "default": "none",
                "tooltip": (
                    "Off: process the entire file in one pass (recommended for most files).\n"
                    "Fixed: split audio into equal-length segments, then join with crossfade.\n"
                    "Smart: split at the quietest point in each window to avoid cutting speech.\n"
                    "Use chunking for very long files to reduce memory usage."
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
                    "Recommended: 30–60 s. Default: 30 s."
                ),
            },
        ]

    def is_available(self) -> bool:
        try:
            from df.enhance import enhance, init_df  # noqa: F401
        except ImportError:
            return False
        return True

    def load(self, progress_cb: Optional[Callable[[str], None]] = None) -> None:
        return None

    def unload(self) -> None:
        return None

    def download(self, model_dir: Path,
                 progress_cb: Optional[Callable[[str], None]] = None) -> None:
        for model_name in ("DeepFilterNet2", "DeepFilterNet3"):
            sub_dir = model_dir / model_name
            if sub_dir.exists() and (sub_dir / "config.ini").exists():
                if progress_cb:
                    progress_cb(f"{model_name} already present, skipping.")
                continue
            _download_model_zip(model_name, sub_dir, progress_cb)
            if progress_cb:
                progress_cb(f"✅ {model_name} ready.")
        if progress_cb:
            progress_cb("✅ All DeepFilterNet models downloaded.")

    def _model_sub_dir(self, model_dir: Path, model_name: str) -> Path:
        sub = model_dir / model_name
        if sub.exists():
            return sub
        return model_dir

    def _prepare_input(self, input_path: str, df_sr: int,
                       progress_cb: Optional[Callable[[str], None]] = None) -> tuple:
        import subprocess
        src_ext = Path(input_path).suffix.lower()
        needs_convert = src_ext not in (".wav", ".flac")
        tmp_wav: Optional[str] = None

        if needs_convert:
            if progress_cb:
                progress_cb("DeepFilterNet: converting input to WAV…")
            fd, tmp_wav = tempfile.mkstemp(suffix=".wav", prefix="dfn_input_")
            os.close(fd)
            r = subprocess.run(
                ["ffmpeg", "-y", "-i", input_path, "-ac", "1", "-ar", str(df_sr), tmp_wav],
                capture_output=True, timeout=300,
            )
            if r.returncode != 0:
                try:
                    Path(tmp_wav).unlink()
                except Exception:
                    pass
                raise RuntimeError(
                    f"ffmpeg WAV conversion failed:\n"
                    f"{r.stderr.decode(errors='replace')[-400:]}")
            return tmp_wav, tmp_wav

        info = sf.info(input_path)
        if info.samplerate != df_sr or info.channels > 1:
            if progress_cb:
                progress_cb(f"DeepFilterNet: resampling {info.samplerate} Hz → {df_sr} Hz…")
            fd, tmp_wav = tempfile.mkstemp(suffix=".wav", prefix="dfn_resamp_")
            os.close(fd)
            r = subprocess.run(
                ["ffmpeg", "-y", "-i", input_path, "-ac", "1", "-ar", str(df_sr), tmp_wav],
                capture_output=True, timeout=300,
            )
            if r.returncode != 0:
                try:
                    Path(tmp_wav).unlink()
                except Exception:
                    pass
                raise RuntimeError(
                    f"ffmpeg resampling failed:\n"
                    f"{r.stderr.decode(errors='replace')[-400:]}")
            return tmp_wav, tmp_wav

        return input_path, None

    def process(self, request: EnhancementRequest,
                progress_cb: Optional[Callable[[str], None]] = None,
                log_cb: Optional[Callable[[str], None]] = None) -> EnhancementResult:
        try:
            from df.enhance import enhance, init_df, load_audio, save_audio
        except ImportError as _ie:
            msg = str(_ie)
            if "df" in msg or "No module named" in msg:
                raise RuntimeError(
                    "deepfilternet is not installed.\n"
                    "Run install/deepfilternet_install.bat or:\n"
                    "  pip install deepfilternet")
            raise RuntimeError(
                f"deepfilternet failed to import due to a missing dependency:\n{msg}\n\n"
                "Re-run install/deepfilternet_install.bat to reinstall all dependencies.")

        import torch

        model_name = str(request.params.get("df_model", "DeepFilterNet3"))
        atten_lim = request.params.get("atten_lim_db", 0)
        atten_lim_db = float(atten_lim) if atten_lim else None
        post_filter = bool(request.params.get("post_filter", False))
        chunk_mode = request.params.get("chunk_mode", "none")
        chunk_sec = float(request.params.get("chunk_sec", 30.0))

        base_model_dir = Path(request.params.get("model_dir", ""))
        sub_dir = self._model_sub_dir(base_model_dir, model_name)

        use_local = sub_dir.exists() and (sub_dir / "config.ini").exists()
        model_base_dir_arg = str(sub_dir) if use_local else None

        if not use_local and progress_cb:
            progress_cb(
                f"DeepFilterNet: model not found locally, downloading {model_name}…\n"
                "(Use 'Download Model' button to pre-download to the models folder)")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        if progress_cb:
            progress_cb(f"DeepFilterNet: loading {model_name} on {device.upper()}…")

        model, df_state, _ = init_df(
            model_base_dir=model_base_dir_arg,
            post_filter=post_filter,
            log_level="WARNING",
            log_file=None,
            config_allow_defaults=True,
            default_model=model_name,
        )

        df_sr = df_state.sr()
        input_path = request.input_path
        output_dir = Path(request.output_dir)
        stem = Path(input_path).stem
        out_path = str(output_dir / f"dfn_{stem}.wav")

        actual_input, tmp_wav = self._prepare_input(input_path, df_sr, progress_cb)

        try:
            if chunk_mode == "none":
                if progress_cb:
                    progress_cb(f"DeepFilterNet: running {model_name} noise suppression…")
                audio, _ = load_audio(actual_input, sr=df_sr)
                enhanced = enhance(
                    model,
                    df_state,
                    audio,
                    pad=True,
                    atten_lim_db=atten_lim_db,
                )
                save_audio(out_path, enhanced, df_sr)
                data, sr_out = sf.read(out_path, dtype="float32")
                duration = len(data) / sr_out if sr_out else 0.0

            else:
                raw, _ = sf.read(actual_input, dtype="float32")
                if raw.ndim > 1:
                    raw = raw.mean(axis=1)

                chunks = (
                    _chunk_fixed(raw, df_sr, chunk_sec)
                    if chunk_mode == "fixed"
                    else _chunk_smart(raw, df_sr, chunk_sec)
                )

                tmp_dir = Path(tempfile.mkdtemp(prefix="dfn_chunks_"))
                try:
                    enhanced_chunks = []
                    for i, chunk in enumerate(chunks):
                        if progress_cb:
                            progress_cb(f"DeepFilterNet: chunk {i + 1}/{len(chunks)}…")
                        chunk_path = str(tmp_dir / f"chunk_{i:04d}.wav")
                        sf.write(chunk_path, chunk, df_sr)
                        audio, _ = load_audio(chunk_path, sr=df_sr)
                        enh = enhance(
                            model,
                            df_state,
                            audio,
                            pad=True,
                            atten_lim_db=atten_lim_db,
                        )
                        enh_np = enh.squeeze(0).numpy() if hasattr(enh, "numpy") else np.array(enh).squeeze()
                        enhanced_chunks.append(enh_np.astype(np.float32))
                finally:
                    shutil.rmtree(str(tmp_dir), ignore_errors=True)

                if progress_cb:
                    progress_cb("DeepFilterNet: assembling chunks with crossfade…")
                joined = _crossfade_concat(enhanced_chunks, df_sr, fade_ms=30)
                sf.write(out_path, joined, df_sr)
                duration = len(joined) / df_sr

        finally:
            if tmp_wav is not None:
                try:
                    Path(tmp_wav).unlink()
                except Exception:
                    pass

        if progress_cb:
            progress_cb(f"✅ DeepFilterNet complete: dfn_{stem}.wav")
        return EnhancementResult(output_path=out_path, sample_rate=df_sr, duration_s=duration)
