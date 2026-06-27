# reuse.py
import os
import re
import shutil
import subprocess
import sys
import time
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


class _CmdResult:
    def __init__(self, returncode: int, stdout_text: str):
        self.returncode = returncode
        self.stdout_text = stdout_text


@register_backend
class ReuseBackend(AudioEnhancerBackend):
    @property
    def name(self) -> str:
        return "RE-USE"

    @property
    def model_id(self) -> str:
        return "reuse"

    @property
    def display_name(self) -> str:
        return "RE-USE Speech Enhancement"

    @property
    def venv_names(self) -> List[str]:
        return ["venv_reuse"]

    @property
    def download_repo(self) -> str:
        return "nvidia/RE-USE"

    @property
    def download_size(self) -> str:
        return "~100 MB"

    @property
    def header_icon(self) -> str:
        return "⚡"

    @property
    def header_title(self) -> str:
        return "RE-USE Speech Enhancement"

    @property
    def requires_gpu(self) -> bool:
        return True

    @property
    def auth_required(self) -> bool:
        return False

    @property
    def supports_demucs_preprocessing(self) -> bool:
        return True

    @property
    def supports_bandwidth_extension(self) -> bool:
        return True

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
                    "Off: process the entire file in one pass (recommended for short files).\n"
                    "Fixed: split audio into equal-length segments, then join with crossfade.\n"
                    "Smart: split at the quietest point in each window to avoid cutting speech.\n"
                    "Use chunking for long files to avoid running out of GPU memory."
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
                    "Longer chunks sound more natural but may run out of VRAM.\n"
                    "Recommended: 15–30 s. Default: 15 s."
                ),
            },
            {
                "key": "bwe",
                "type": "choice",
                "label": "Bandwidth Extension",
                "options": [
                    (0, "Disabled (preserve original SR)"),
                    (8000, "→ 8 kHz"),
                    (16000, "→ 16 kHz"),
                    (22050, "→ 22 kHz"),
                    (24000, "→ 24 kHz"),
                    (32000, "→ 32 kHz"),
                    (44100, "→ 44.1 kHz"),
                    (48000, "→ 48 kHz (max)"),
                ],
                "default": 0,
                "tooltip": (
                    "Bandwidth Extension (BWE): upsample the output to a higher sample rate\n"
                    "using the model's built-in neural BWE module.\n"
                    "Disabled: output keeps the original sample rate of the input.\n"
                    "Selecting a target SR synthesises the missing high-frequency content.\n"
                    "Recommended for telephony (8/16 kHz) input to reach 44.1 or 48 kHz output."
                ),
            },
        ]

    def is_available(self) -> bool:
        try:
            import torch  # noqa: F401
        except ImportError:
            return False
        try:
            import mamba_ssm  # noqa: F401
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
                "Run install/reuse_install.bat or: pip install huggingface_hub")
        model_dir.mkdir(parents=True, exist_ok=True)
        if progress_cb:
            progress_cb(f"Downloading {self.download_repo} from HuggingFace…  "
                         f"(≈ {self.download_size} — please wait)")
        snapshot_download(
            repo_id=self.download_repo,
            local_dir=str(model_dir),
            local_dir_use_symlinks=False,
        )
        if progress_cb:
            progress_cb("✅ Model downloaded successfully.")

    # ── inference script discovery / invocation ─────────────────────────────
    def _find_script(self, model_dir: Path, names: List[str]) -> Optional[Path]:
        for name in names:
            p = model_dir / name
            if p.exists():
                return p
        for name in names:
            found = list(model_dir.rglob(name))
            if found:
                return found[0]
        return None

    def _parse_sh(self, sh_path: Path, parsed_env: dict) -> Optional[List[str]]:
        try:
            text = sh_path.read_text(encoding="utf-8", errors="replace")
            for line in text.splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if re.search(r"\bpython\b", line, re.I):
                    line = re.sub(r"\s*\\$", "", line).split("#")[0].strip()
                    parts = line.split()
                    cmd_parts: List[str] = []
                    for tok in parts:
                        env_m = re.match(r'^([A-Z_][A-Z0-9_]*)=(.*)', tok)
                        if env_m and not cmd_parts:
                            parsed_env[env_m.group(1)] = env_m.group(2).strip("'\"")
                        else:
                            if re.fullmatch(r"python[\d.]*", tok, re.I):
                                tok = sys.executable
                            cmd_parts.append(tok)
                    return cmd_parts if cmd_parts else None
        except Exception:
            pass
        return None

    def _to_wav(self, src: str, dest: str) -> None:
        r = subprocess.run(
            ["ffmpeg", "-y", "-i", src, "-ac", "1", dest],
            capture_output=True, timeout=300,
        )
        if r.returncode != 0:
            raise RuntimeError(
                f"ffmpeg WAV conversion failed:\n{r.stderr.decode(errors='replace')[-400:]}")

    def _find_model_config(self, md: Path) -> Optional[Path]:
        import yaml

        def candidates():
            priority = ["config.yaml", "config.yml", "model_config.yaml",
                        "model_config.yml", "config.json"]
            for name in priority:
                p = md / name
                if p.exists():
                    yield p
            for pat in ("*.yaml", "*.yml", "*.json"):
                for p in sorted(md.rglob(pat)):
                    if p.name.lower() in ("hparams.yaml", "hparams.yml"):
                        continue
                    yield p

        for p in candidates():
            try:
                with open(p, encoding="utf-8") as f:
                    if p.suffix in (".yaml", ".yml"):
                        data = yaml.safe_load(f)
                    else:
                        import json
                        data = json.load(f)
                if isinstance(data, dict) and "stft_cfg" in data:
                    return p
            except Exception:
                continue
        return None

    def _run_cmd(self, cmd: List[str], cwd: str, parsed_env: dict,
                 log_cb: Optional[Callable[[str], None]],
                 progress_cb: Optional[Callable[[str], None]]) -> _CmdResult:
        env = {**os.environ, "PYTHONPATH": cwd}
        if parsed_env:
            env.update(parsed_env)
        if log_cb:
            log_cb(f"$ {' '.join(cmd)}\n")
        proc = subprocess.Popen(
            cmd, cwd=cwd,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            env=env, text=True, errors="replace",
        )
        output_lines: List[str] = []
        for line in proc.stdout:
            output_lines.append(line)
            if log_cb:
                log_cb(line)
            stripped = line.strip()
            if stripped and len(stripped) < 200 and progress_cb:
                progress_cb(stripped)
        proc.wait()
        return _CmdResult(proc.returncode, "".join(output_lines))

    def _build_inference_cmd(self, noisy_dir: Path, enhanced_dir: Path, md: Path,
                              bwe: int, parsed_env: dict,
                              log_cb: Optional[Callable[[str], None]]) -> List[str]:
        sh_path = self._find_script(md, ["inference.sh"])
        cmd = None
        if sh_path:
            cmd = self._parse_sh(sh_path, parsed_env)
            if log_cb:
                log_cb(f"[info] Parsed from {sh_path.name}: {cmd}\n")

        if not cmd:
            script = self._find_script(
                md, ["inference.py", "enhance.py", "run_enhancement.py",
                     "run_enhance.py", "infer.py", "main.py"])
            if not script:
                raise RuntimeError(
                    "Could not locate an inference script in the downloaded model repo.\n"
                    "Expected one of: inference.py / enhance.py / run_enhancement.py\n"
                    f"Model directory: {md}\n"
                    f"Contents: {[p.name for p in md.iterdir() if not p.name.startswith('.')]}")
            cmd = [sys.executable, str(script)]
            if log_cb:
                log_cb(f"[info] Fallback script: {script}\n")

        cmd_str = " ".join(str(c) for c in cmd)

        if "--input_folder" not in cmd_str and "--input_dir" not in cmd_str \
                and "--noisy_dir" not in cmd_str and "--input_path" not in cmd_str:
            in_flag = "--input_folder"
            script_path = Path(cmd[1]) if len(cmd) > 1 and Path(cmd[1]).exists() else None
            if script_path:
                try:
                    src_text = script_path.read_text(encoding="utf-8", errors="replace")
                    for candidate in ("--input_dir", "--noisy_dir", "--input_path",
                                      "--input_folder", "--noisy_folder"):
                        if candidate in src_text:
                            in_flag = candidate
                            break
                except Exception:
                    pass
            if log_cb:
                log_cb(f"[info] Using input flag: {in_flag}\n")
            cmd.extend([in_flag, str(noisy_dir)])

        if "--output_folder" not in cmd_str and "--output_dir" not in cmd_str \
                and "--output_path" not in cmd_str and "--out_dir" not in cmd_str:
            out_flag = "--output_folder"
            script_path = Path(cmd[1]) if len(cmd) > 1 and Path(cmd[1]).exists() else None
            if script_path:
                try:
                    src_text = script_path.read_text(encoding="utf-8", errors="replace")
                    for candidate in ("--output_dir", "--out_dir", "--output_path",
                                      "--output_folder", "--enhanced_dir"):
                        if candidate in src_text:
                            out_flag = candidate
                            break
                except Exception:
                    pass
            if log_cb:
                log_cb(f"[info] Using output flag: {out_flag}\n")
            cmd.extend([out_flag, str(enhanced_dir)])

        if "--checkpoint_file" not in cmd_str:
            ckpt_exts = (".ckpt", ".pt", ".pth", ".bin", ".safetensors")
            ckpt = None
            for ext in ckpt_exts:
                found = sorted(md.rglob(f"*{ext}"))
                if found:
                    ckpt = found[0]
                    break
            if ckpt is None:
                raise RuntimeError(
                    "No checkpoint file found in the model directory.\n"
                    f"Searched for: {ckpt_exts}\n"
                    f"Model directory: {md}\n"
                    "Try re-downloading the model via the 'Download Model' button.")
            if log_cb:
                log_cb(f"[info] Checkpoint: {ckpt}\n")
            cmd.extend(["--checkpoint_file", str(ckpt)])

        if "--config" not in cmd_str:
            cfg = self._find_model_config(md)
            if cfg:
                if log_cb:
                    log_cb(f"[info] Config: {cfg}\n")
                cmd.extend(["--config", str(cfg)])
            else:
                if log_cb:
                    log_cb("[info] No valid config found, letting inference.py use its default.\n")

        if bwe and "--BWE" not in cmd_str:
            cmd.extend(["--BWE", str(bwe)])

        if log_cb:
            log_cb(f"[info] Command: {cmd}\n")
            log_cb(f"[info] CWD: {md}\n")
        return cmd

    def _collect_output(self, enhanced_dir: Path, in_name: str) -> Path:
        out_files = [
            p for p in enhanced_dir.rglob("*")
            if p.suffix.lower() in (".wav", ".flac") and p.is_file()
        ]
        if not out_files:
            raise RuntimeError(
                f"No output file produced in {enhanced_dir}\n"
                "The model ran but created no audio file.\n"
                "Check the terminal for inference output above.")
        stem = Path(in_name).stem
        matched = [f for f in out_files if f.stem == stem]
        return matched[0] if matched else out_files[0]

    def process(self, request: EnhancementRequest,
                progress_cb: Optional[Callable[[str], None]] = None,
                log_cb: Optional[Callable[[str], None]] = None) -> EnhancementResult:
        model_dir = Path(request.params["model_dir"])
        bwe = int(request.params.get("bwe", 0) or 0)
        chunk_mode = request.params.get("chunk_mode", "none")
        chunk_sec = float(request.params.get("chunk_sec", 15.0))
        output_dir = Path(request.output_dir)

        md = model_dir
        noisy_dir = md / "noisy_audio"
        enhanced_dir = md / "enhanced_audio"
        noisy_dir.mkdir(exist_ok=True)
        enhanced_dir.mkdir(exist_ok=True)

        def _clear_dirs():
            for f in noisy_dir.iterdir():
                if f.is_file():
                    try:
                        f.unlink()
                    except Exception:
                        pass
            for f in list(enhanced_dir.glob("*.wav")) + list(enhanced_dir.glob("*.flac")):
                try:
                    f.unlink()
                except Exception:
                    pass

        _clear_dirs()

        src = request.input_path
        ext = Path(src).suffix.lower()
        tmp_wav = noisy_dir / (Path(src).stem + ".wav")

        if ext != ".wav":
            if progress_cb:
                progress_cb("Converting input to mono WAV…")
            self._to_wav(src, str(tmp_wav))
        else:
            try:
                info = sf.info(src)
                channels = info.channels
            except Exception:
                try:
                    r = subprocess.run(
                        ["ffprobe", "-v", "quiet", "-select_streams", "a:0",
                         "-show_entries", "stream=channels", "-of", "csv=p=0", src],
                        capture_output=True, timeout=10, text=True)
                    channels = int(r.stdout.strip()) if r.returncode == 0 and r.stdout.strip() else 2
                except Exception:
                    channels = 2

            if channels > 1:
                if progress_cb:
                    progress_cb("Converting stereo input to mono…")
                proc = subprocess.run(
                    ["ffmpeg", "-y", "-i", src, "-ac", "1", str(tmp_wav)],
                    capture_output=True, timeout=300)
                if proc.returncode != 0:
                    raise RuntimeError(
                        f"ffmpeg mono conversion failed:\n"
                        f"{proc.stderr.decode(errors='replace')[-500:]}")
            else:
                if tmp_wav.resolve() != Path(src).resolve():
                    shutil.copy2(src, str(tmp_wav))

        try:
            import torch as _t
        except ImportError:
            raise RuntimeError(
                "PyTorch is not installed.\n"
                "Run: pip install torch torchaudio")
        if not _t.cuda.is_available():
            raise RuntimeError(
                "No CUDA GPU detected.\n"
                "The RE-USE inference.py requires an NVIDIA GPU with CUDA.\n"
                "CPU-only mode is not supported by this model.")
        if log_cb:
            log_cb(f"[info] CUDA device: {_t.cuda.get_device_name(0)}\n")

        if progress_cb:
            progress_cb("Running RE-USE inference…")

        parsed_env: dict = {}
        cmd = self._build_inference_cmd(noisy_dir, enhanced_dir, md, bwe, parsed_env, log_cb)

        if chunk_mode == "none":
            result = self._run_cmd(cmd, str(md), parsed_env, log_cb, progress_cb)
            if result.returncode != 0:
                tail = result.stdout_text.strip()[-2000:] if result.stdout_text.strip() else "(no output captured)"
                raise RuntimeError(
                    f"Inference process exited with code {result.returncode}.\n\n"
                    f"--- subprocess output (last 2000 chars) ---\n{tail}")
            src_out = self._collect_output(enhanced_dir, tmp_wav.name)
            stem = tmp_wav.stem
            final = output_dir / f"enhanced_{stem}{src_out.suffix}"
            shutil.copy2(str(src_out), str(final))
            if progress_cb:
                progress_cb(f"✅ RE-USE complete: {final.name}")
            data, sr = sf.read(str(final), dtype="float32")
            duration = len(data) / sr if sr else 0.0
            return EnhancementResult(output_path=str(final), sample_rate=sr, duration_s=duration)

        audio, sr = sf.read(str(tmp_wav), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        chunks = (chunk_fixed(audio, sr, chunk_sec)
                  if chunk_mode == "fixed"
                  else chunk_smart(audio, sr, chunk_sec))

        if progress_cb:
            progress_cb(f"Split into {len(chunks)} chunks, processing…")
        enhanced_chunks = []
        out_sr = sr

        for i, chunk in enumerate(chunks):
            if progress_cb:
                progress_cb(f"Processing chunk {i + 1}/{len(chunks)}…")
            _clear_dirs()
            chunk_name = f"chunk_{i:04d}.wav"
            sf.write(str(noisy_dir / chunk_name), chunk, sr)
            result = self._run_cmd(cmd, str(md), parsed_env, log_cb, progress_cb)
            if result.returncode != 0:
                tail = result.stdout_text.strip()[-2000:] if result.stdout_text.strip() else "(no output)"
                raise RuntimeError(
                    f"Chunk {i + 1} inference failed (code {result.returncode}).\n\n"
                    f"--- output ---\n{tail}")
            src_out = self._collect_output(enhanced_dir, chunk_name)
            enh, enh_sr = sf.read(str(src_out), dtype="float32")
            if enh.ndim > 1:
                enh = enh.mean(axis=1)
            if i == 0:
                out_sr = enh_sr
            enhanced_chunks.append(enh)

        if progress_cb:
            progress_cb("Assembling chunks with crossfade…")
        stem = tmp_wav.stem
        final = output_dir / f"enhanced_{stem}.wav"
        joined = crossfade_concat(enhanced_chunks, out_sr, fade_ms=30)
        sf.write(str(final), joined, out_sr)
        if progress_cb:
            progress_cb(f"✅ RE-USE complete: {final.name}")
        duration = len(joined) / out_sr if out_sr else 0.0
        return EnhancementResult(output_path=str(final), sample_rate=out_sr, duration_s=duration)
