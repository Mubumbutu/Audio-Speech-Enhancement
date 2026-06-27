# clearervoice_backend.py
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
)

_CV_TASK_SR   = "speech_super_resolution"
_CV_TASK_SE   = "speech_enhancement"
_MODEL_SR     = "MossFormer2_SR_48K"
_MODEL_SE_48K = "MossFormer2_SE_48K"
_MODEL_SE_16K = "FRCRN_SE_16K"
_MODEL_SE_GAN = "MossFormerGAN_SE_16K"

_HF_REPOS = {
    _MODEL_SR:     "alibabasglab/MossFormer2_SR_48K",
    _MODEL_SE_48K: "alibabasglab/MossFormer2_SE_48K",
    _MODEL_SE_16K: "alibabasglab/FRCRN_SE_16K",
    _MODEL_SE_GAN: "alibabasglab/MossFormerGAN_SE_16K",
}

_TASK_FOR_MODEL = {
    _MODEL_SR:     _CV_TASK_SR,
    _MODEL_SE_48K: _CV_TASK_SE,
    _MODEL_SE_16K: _CV_TASK_SE,
    _MODEL_SE_GAN: _CV_TASK_SE,
}

_INPUT_SR_FOR_MODEL = {
    _MODEL_SR:     16000,
    _MODEL_SE_48K: 48000,
    _MODEL_SE_16K: 16000,
    _MODEL_SE_GAN: 16000,
}


def _make_junction_win(link: Path, target: Path) -> None:
    if link.exists() or link.is_symlink():
        return
    link.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(link), str(target)],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _ensure_cv_checkpoints_link(app_dir: Path, models_base: Path) -> None:
    link_dir = app_dir / "clearvoice" / "checkpoints"
    if not link_dir.exists():
        link_dir.parent.mkdir(parents=True, exist_ok=True)
        models_base.mkdir(parents=True, exist_ok=True)
        try:
            if os.name == "nt":
                _make_junction_win(link_dir, models_base)
            else:
                link_dir.symlink_to(models_base)
        except Exception:
            link_dir.mkdir(parents=True, exist_ok=True)


def _prepare_input_wav(input_path: str, target_sr: int, tmp_dir: Path) -> str:
    try:
        import librosa
        audio, _ = librosa.load(input_path, sr=target_sr, mono=True)
    except ImportError:
        audio, src_sr = sf.read(input_path, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if src_sr != target_sr:
            try:
                import resampy
                audio = resampy.resample(audio, src_sr, target_sr)
            except ImportError:
                from scipy.signal import resample_poly
                import math
                g = math.gcd(target_sr, src_sr)
                audio = resample_poly(audio, target_sr // g, src_sr // g).astype(np.float32)

    tmp_path = str(tmp_dir / "cv_input.wav")
    sf.write(tmp_path, audio.astype(np.float32), target_sr)
    return tmp_path


@register_backend
class ClearerVoiceBackend(AudioEnhancerBackend):
    @property
    def name(self) -> str:
        return "ClearerVoice"

    @property
    def model_id(self) -> str:
        return "clearervoice"

    @property
    def display_name(self) -> str:
        return "ClearerVoice Studio"

    @property
    def venv_names(self) -> List[str]:
        return ["venv_clearervoice"]

    @property
    def download_repo(self) -> str:
        return "alibabasglab/MossFormer2_SR_48K"

    @property
    def download_size(self) -> str:
        return "~270 MB"

    @property
    def header_icon(self) -> str:
        return "🔊"

    @property
    def header_title(self) -> str:
        return "ClearerVoice Studio"

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
                "key": "cv_task",
                "type": "choice",
                "label": "Task",
                "options": [
                    (_CV_TASK_SR, "Super-Resolution (16→48 kHz)"),
                    (_CV_TASK_SE, "Speech Enhancement (denoising)"),
                ],
                "default": _CV_TASK_SR,
                "tooltip": (
                    "Super-Resolution: converts low-resolution audio (≥16 kHz effective)\n"
                    "                  to 48 kHz by reconstructing missing high frequencies.\n"
                    "                  Best for clean recordings with low sample rate.\n"
                    "Speech Enhancement: removes background noise from speech.\n"
                    "                  Works at 16 kHz or 48 kHz depending on model chosen."
                ),
            },
            {
                "key": "cv_model",
                "type": "choice",
                "label": "Model",
                "options": [
                    (_MODEL_SR,     "MossFormer2_SR_48K   — super-resolution  (~270 MB)"),
                    (_MODEL_SE_48K, "MossFormer2_SE_48K   — enhancement 48 kHz (~130 MB)"),
                    (_MODEL_SE_16K, "FRCRN_SE_16K         — enhancement 16 kHz (~50 MB)"),
                    (_MODEL_SE_GAN, "MossFormerGAN_SE_16K — enhancement 16 kHz (~110 MB)"),
                ],
                "default": _MODEL_SR,
                "tooltip": (
                    "MossFormer2_SR_48K:      Speech super-resolution. Output always 48 kHz.\n"
                    "MossFormer2_SE_48K:      Fullband speech enhancement. Output 48 kHz.\n"
                    "FRCRN_SE_16K:            16 kHz noise suppression. Output 16 kHz.\n"
                    "MossFormerGAN_SE_16K:    GAN-based 16 kHz enhancement. Output 16 kHz."
                ),
            },
        ]

    def _model_checkpoint_dir(self, model_dir: Path, model_name: str) -> Path:
        return model_dir / model_name

    def is_available(self) -> bool:
        try:
            from clearvoice import ClearVoice  # noqa: F401
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
                "Run install/clearervoice_install.bat to set up the environment.")

        model_dir.mkdir(parents=True, exist_ok=True)

        for model_name, repo_id in _HF_REPOS.items():
            dest = self._model_checkpoint_dir(model_dir, model_name)
            if dest.exists() and any(dest.iterdir()):
                if progress_cb:
                    progress_cb(f"{model_name} already present, skipping.")
                continue

            if progress_cb:
                progress_cb(f"Downloading {model_name} from {repo_id}…")

            dest.mkdir(parents=True, exist_ok=True)
            snapshot_download(
                repo_id=repo_id,
                local_dir=str(dest),
                local_dir_use_symlinks=False,
                ignore_patterns=["*.md", "*.txt", ".gitattributes"],
            )

            if progress_cb:
                progress_cb(f"✅ {model_name} downloaded.")

        if progress_cb:
            progress_cb("✅ All ClearerVoice models ready.")

    def _is_model_downloaded(self, model_dir: Path, model_name: str) -> bool:
        dest = self._model_checkpoint_dir(model_dir, model_name)
        if not dest.exists():
            return False
        return any(
            f.suffix in (".pt", ".pth", ".ckpt", "")
            for f in dest.iterdir()
            if f.is_file()
        )

    def _get_app_dir(self, model_dir: Path) -> Path:
        return model_dir.parent.parent

    def _setup_checkpoint_link(self, model_dir: Path) -> None:
        app_dir = self._get_app_dir(model_dir)
        _ensure_cv_checkpoints_link(app_dir, model_dir)

    def process(self, request: EnhancementRequest,
                progress_cb: Optional[Callable[[str], None]] = None,
                log_cb: Optional[Callable[[str], None]] = None) -> EnhancementResult:
        try:
            from clearvoice import ClearVoice
        except ImportError:
            raise RuntimeError(
                "clearvoice is not installed.\n"
                "Run install/clearervoice_install.bat to set up the environment.")

        model_dir  = Path(request.params.get("model_dir", ""))
        cv_task    = str(request.params.get("cv_task", _CV_TASK_SR))
        cv_model   = str(request.params.get("cv_model", _MODEL_SR))
        input_path = request.input_path
        output_dir = Path(request.output_dir)
        stem       = Path(input_path).stem

        if not model_dir.is_dir():
            raise RuntimeError(
                "Model directory not configured.\n"
                "Click 'Download Model' to fetch ClearerVoice weights first.")

        if not self._is_model_downloaded(model_dir, cv_model):
            raise RuntimeError(
                f"{cv_model} weights not found.\n"
                "Click 'Download Model' to fetch them first.")

        task_for_model = _TASK_FOR_MODEL.get(cv_model, cv_task)
        if task_for_model != cv_task:
            if progress_cb:
                progress_cb(
                    f"Note: switching task to '{task_for_model}' "
                    f"to match model '{cv_model}'.")
            cv_task = task_for_model

        target_sr = _INPUT_SR_FOR_MODEL.get(cv_model, 16000)

        self._setup_checkpoint_link(model_dir)
        app_dir  = self._get_app_dir(model_dir)
        prev_cwd = os.getcwd()

        tmp_dir = output_dir / "cv_tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)

        try:
            if progress_cb:
                progress_cb(
                    f"ClearerVoice: preparing input (mono {target_sr // 1000} kHz WAV)…")

            prepared_path = _prepare_input_wav(input_path, target_sr, tmp_dir)

            os.chdir(str(app_dir))

            if progress_cb:
                progress_cb(
                    f"ClearerVoice: initialising {cv_model} for {cv_task}…")

            cv = ClearVoice(task=cv_task, model_names=[cv_model])

            if progress_cb:
                progress_cb(f"ClearerVoice: processing with {cv_model}…")

            raw_result = cv(
                input_path=prepared_path,
                online_write=False,
            )

        finally:
            os.chdir(prev_cwd)

        import torch

        if isinstance(raw_result, dict):
            if len(raw_result) == 0:
                raise RuntimeError(
                    "ClearerVoice returned an empty result.\n"
                    "This can happen if the model failed to initialise or the input "
                    "file could not be decoded. Check that all model weights are "
                    "downloaded correctly.")
            audio = next(iter(raw_result.values()))
        else:
            audio = raw_result

        if isinstance(audio, (list, tuple)):
            audio = audio[0]

        if isinstance(audio, torch.Tensor):
            audio = audio.cpu().float().numpy()

        audio = np.asarray(audio, dtype=np.float32)

        if audio.ndim == 0 or audio.size == 0:
            raise RuntimeError(
                "ClearerVoice returned an empty audio array.\n"
                "Check that the model weights are downloaded correctly.")

        if audio.ndim > 1:
            audio = audio.squeeze()
        if audio.ndim > 1:
            audio = audio.mean(axis=0)

        out_sr_map = {
            _MODEL_SR:     48000,
            _MODEL_SE_48K: 48000,
            _MODEL_SE_16K: 16000,
            _MODEL_SE_GAN: 16000,
        }
        out_sr = out_sr_map.get(cv_model, 48000)

        out_path = str(output_dir / f"clearervoice_{stem}.wav")
        sf.write(out_path, audio, out_sr)
        duration = len(audio) / out_sr

        try:
            import shutil
            shutil.rmtree(str(tmp_dir), ignore_errors=True)
        except Exception:
            pass

        if progress_cb:
            progress_cb(f"✅ ClearerVoice complete: clearervoice_{stem}.wav")

        return EnhancementResult(
            output_path=out_path,
            sample_rate=out_sr,
            duration_s=duration,
        )