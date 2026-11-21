import os
import subprocess
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
    atomic_temp_path,
    finalize_atomic_write,
    cleanup_atomic_temp,
)

_CHECKPOINTS = {
    "offline": "denoiser.safetensors",
    "streaming": "denoiser_realtime.safetensors",
}


@register_backend
class VoiceIsolatorV1Backend(AudioEnhancerBackend):

    @property
    def name(self) -> str:
        return "Voice Isolator v1"

    @property
    def model_id(self) -> str:
        return "voice_isolator_v1"

    @property
    def display_name(self) -> str:
        return "Inworld Voice Isolator v1"

    @property
    def venv_names(self) -> List[str]:
        return ["venv_voice_isolator", "venv_voice_isolator_v1", "venv_voiceisolator"]

    @property
    def download_repo(self) -> str:
        return "inworld/voice-isolator"

    @property
    def download_size(self) -> str:
        return "~490 MB"

    @property
    def header_icon(self) -> str:
        return "🎙"

    @property
    def header_title(self) -> str:
        return "Inworld Voice Isolator v1"

    @property
    def requires_gpu(self) -> bool:
        return False

    @property
    def auth_required(self) -> bool:
        return False

    @property
    def supports_demucs_preprocessing(self) -> bool:
        return True

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
                "label": "Mode",
                "options": [
                    ("offline", "Offline (best quality)"),
                    ("streaming", "Streaming (realtime model)"),
                ],
                "default": "offline",
                "tooltip": (
                    "Offline: voice-isolator-v1 (64M), two-sided; every frame sees its\n"
                    "whole 8 s chunk, long files render in 8 s chunks overlapping by 2 s.\n"
                    "Streaming: voice-isolator-v1-realtime (58M), causal; the file is\n"
                    "processed exactly as a live session would hear it.\n"
                    "Use Offline for recordings, Streaming to match live-pipeline output."
                ),
            },
            {
                "key": "lookahead_ms",
                "type": "choice",
                "label": "Look-ahead",
                "options": [
                    (0, "0 ms (lowest latency)"),
                    (20, "20 ms (recommended)"),
                    (40, "40 ms (best quality)"),
                ],
                "default": 20,
                "visible_when": {"key": "mode", "value": "streaming"},
                "tooltip": (
                    "Trained look-ahead of the realtime model (10 ms frames).\n"
                    "20 ms gives 63 ms of algorithmic latency in a live pipeline.\n"
                    "Only used in Streaming mode."
                ),
            },
            {
                "key": "offline_passes",
                "type": "spin",
                "label": "Passes",
                "min": 1,
                "max": 4,
                "step": 1,
                "default": 1,
                "visible_when": {"key": "mode", "value": "offline"},
                "tooltip": (
                    "Number of offline renders. Each extra pass runs the model again\n"
                    "over its own output and adds one full render of processing time.\n"
                    "Default: 1 (the released checkpoint's own count). Offline mode only."
                ),
            },
            {
                "key": "device",
                "type": "choice",
                "label": "Device",
                "options": [
                    ("auto", "Auto (CUDA / Metal / CPU)"),
                    ("cuda", "CUDA (NVIDIA GPU)"),
                    ("mps", "Metal (Apple Silicon)"),
                    ("cpu", "CPU"),
                ],
                "default": "auto",
                "tooltip": (
                    "Auto: CUDA when available, then Apple Metal (PyTorch 2.6+), then CPU.\n"
                    "CPU works everywhere but is slower.\n"
                    "Tested on NVIDIA Ampere, Hopper and Blackwell GPUs."
                ),
            },
            {
                "key": "precision",
                "type": "choice",
                "label": "Precision",
                "options": [
                    ("auto", "Auto (bf16 CUDA / fp32 else)"),
                    ("fp32", "fp32 (highest quality)"),
                    ("bf16", "bf16 (lower VRAM)"),
                    ("fp16", "fp16 (lowest VRAM)"),
                ],
                "default": "auto",
                "tooltip": (
                    "Auto: bfloat16 autocast on CUDA (the training precision),\n"
                    "float32 elsewhere. STFT, iSTFT and the selective scan\n"
                    "always run in float32."
                ),
            },
            {
                "key": "output_sr",
                "type": "choice",
                "label": "Output Sample Rate",
                "options": [
                    (48000, "48 kHz (model native)"),
                    (0, "Keep original input rate"),
                ],
                "default": 48000,
                "tooltip": (
                    "48 kHz: the model's native rate (recommended).\n"
                    "Keep original: resample the output back to the input file's rate\n"
                    "with the package's high-quality kaiser_best resampler.\n"
                    "This model denoises only; it does not restore missing bandwidth."
                ),
            },
            {
                "key": "background",
                "type": "bool",
                "label": "Save removed background (input − voice)",
                "default": False,
                "tooltip": (
                    "Also write voice_isolator_<name>_background.wav (float32) into the\n"
                    "output folder: everything the model removed (noise, music, other voices).\n"
                    "Voice + background reconstruct the aligned mono input.\n"
                    "The background can contain voice leakage."
                ),
            },
        ]

    def is_available(self) -> bool:
        try:
            from inworld import voice_isolation  # noqa: F401
        except ImportError:
            return False
        return True

    def load(self, progress_cb: Optional[Callable[[str], None]] = None) -> None:
        return None

    def unload(self) -> None:
        self._model = None
        self._cache_key = None
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def download(self, model_dir: Path,
                 progress_cb: Optional[Callable[[str], None]] = None) -> None:
        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            raise RuntimeError(
                "huggingface_hub is not installed.\n"
                "Run: pip install huggingface_hub")
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
            progress_cb("✅ Voice Isolator models downloaded successfully.")

    def _load_model(self, model_dir: Path, mode: str, lookahead_frames: int,
                    device: str, precision: str, offline_passes: int,
                    progress_cb: Optional[Callable[[str], None]],
                    log_cb: Optional[Callable[[str], None]]):
        try:
            from inworld import voice_isolation
        except ImportError:
            raise RuntimeError(
                "inworld-voice-isolation is not installed.\n"
                "Run: pip install inworld-voice-isolation\n"
                "On Linux with CUDA: pip install \"inworld-voice-isolation[fast]\"")
        ckpt = model_dir / _CHECKPOINTS[mode]
        if ckpt.is_file():
            model_ref = str(ckpt)
        else:
            model_ref = self.download_repo
            if progress_cb:
                progress_cb(
                    "Checkpoint not found locally — downloading from "
                    "HuggingFace (checksum verified)…")
        key = (model_ref, mode, lookahead_frames, device, precision, offline_passes)
        if getattr(self, "_model", None) is not None \
                and getattr(self, "_cache_key", None) == key:
            return self._model
        self.unload()
        if log_cb:
            log_cb(f"[info] Checkpoint: {model_ref}\n")
        if progress_cb:
            progress_cb(f"Loading Voice Isolator v1 ({mode})…")
        kwargs = dict(
            mode=mode,
            precision=precision,
            lookahead_frames=lookahead_frames if mode == "streaming" else 0,
            offline_passes=offline_passes if mode == "offline" else None,
        )
        if device != "auto":
            kwargs["device"] = device
        model = voice_isolation.VoiceIsolator(model_ref, **kwargs)
        if log_cb:
            log_cb(f"[info] Device: {model.device} · precision: {model.precision}\n")
        self._model = model
        self._cache_key = key
        return model

    def _ffmpeg_to_wav(self, src: str) -> str:
        fd, tmp = tempfile.mkstemp(suffix=".wav", prefix="voice_isolator_")
        os.close(fd)
        r = subprocess.run(
            ["ffmpeg", "-y", "-i", src, "-ac", "1", tmp],
            capture_output=True, timeout=600,
        )
        if r.returncode != 0:
            try:
                os.remove(tmp)
            except OSError:
                pass
            raise RuntimeError(
                f"ffmpeg decoding failed:\n{r.stderr.decode(errors='replace')[-400:]}")
        return tmp

    def process(self, request: EnhancementRequest,
                progress_cb: Optional[Callable[[str], None]] = None,
                log_cb: Optional[Callable[[str], None]] = None) -> EnhancementResult:
        model_dir    = Path(request.params.get("model_dir", ""))
        mode         = str(request.params.get("mode", "offline"))
        lookahead_ms = int(request.params.get("lookahead_ms", 20))
        device       = str(request.params.get("device", "auto"))
        precision    = str(request.params.get("precision", "auto"))
        output_sr    = int(request.params.get("output_sr", 48000))
        save_bg      = bool(request.params.get("background", False))
        passes       = int(request.params.get("offline_passes", 1))

        if mode not in _CHECKPOINTS:
            raise RuntimeError(f"Unknown mode: {mode!r} (expected offline or streaming).")

        input_path = request.input_path
        output_dir = Path(request.output_dir)
        stem       = Path(input_path).stem

        model = self._load_model(
            model_dir, mode, lookahead_ms // 10, device, precision, passes,
            progress_cb, log_cb)

        import torch
        from inworld import voice_isolation

        tmp_wav = None
        try:
            try:
                wav, in_sr = voice_isolation.load_audio(input_path)
            except Exception:
                if log_cb:
                    log_cb("[info] soundfile could not read the input, decoding with ffmpeg…\n")
                tmp_wav = self._ffmpeg_to_wav(input_path)
                wav, in_sr = voice_isolation.load_audio(tmp_wav)
            if not wav.numel():
                raise RuntimeError("The input audio is empty.")

            out_sr = output_sr if output_sr else in_sr
            if log_cb:
                log_cb(f"[info] Input: {in_sr} Hz · output: {out_sr} Hz · mode: {mode}\n")
            if progress_cb:
                progress_cb(
                    f"Voice Isolator: isolating {len(wav) / max(1, in_sr):.1f} s "
                    f"of audio ({mode})…")

            voice = model.isolate(wav, in_sr, output_sample_rate=out_sr)
            voice_np = torch.as_tensor(voice).detach().cpu().numpy()
            voice_np = np.ascontiguousarray(voice_np, dtype=np.float32).reshape(-1)

            out_path = str(output_dir / f"voice_isolator_{stem}.wav")
            tmp_out = atomic_temp_path(out_path)
            try:
                sf.write(tmp_out, voice_np, out_sr, subtype="FLOAT")
                finalize_atomic_write(tmp_out, out_path)
            except BaseException:
                cleanup_atomic_temp(tmp_out)
                raise

            if save_bg:
                if progress_cb:
                    progress_cb("Voice Isolator: saving removed background…")
                if out_sr != in_sr:
                    ref = voice_isolation.resample(wav, in_sr, 48000)
                    ref = voice_isolation.resample(ref, 48000, out_sr)
                else:
                    ref = wav
                ref_np = torch.as_tensor(ref).detach().cpu().numpy()
                ref_np = np.ascontiguousarray(ref_np, dtype=np.float32).reshape(-1)
                n = min(len(ref_np), len(voice_np))
                bg = ref_np[:n] - voice_np[:n]
                bg_path = str(output_dir / f"voice_isolator_{stem}_background.wav")
                tmp_bg = atomic_temp_path(bg_path)
                try:
                    sf.write(tmp_bg, bg, out_sr, subtype="FLOAT")
                    finalize_atomic_write(tmp_bg, bg_path)
                except BaseException:
                    cleanup_atomic_temp(tmp_bg)
                    raise
                if log_cb:
                    log_cb(f"[info] Background: {bg_path}\n")

            duration = len(voice_np) / out_sr if out_sr else 0.0
            if progress_cb:
                progress_cb(f"✅ Voice Isolator complete: voice_isolator_{stem}.wav")
            return EnhancementResult(
                output_path=out_path, sample_rate=out_sr, duration_s=duration)
        finally:
            if tmp_wav:
                try:
                    os.remove(tmp_wav)
                except OSError:
                    pass