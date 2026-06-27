# models_backends.py
import os
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type


@dataclass
class EnhancementRequest:
    input_path: str
    output_dir: str
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EnhancementResult:
    output_path: str
    sample_rate: int
    duration_s: float


class AudioEnhancerBackend(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @property
    @abstractmethod
    def model_id(self) -> str:
        ...

    @property
    @abstractmethod
    def display_name(self) -> str:
        ...

    @property
    @abstractmethod
    def venv_names(self) -> List[str]:
        ...

    @property
    def download_repo(self) -> str:
        return ""

    @property
    def download_size(self) -> str:
        return ""

    @property
    def header_icon(self) -> str:
        return "⚡"

    @property
    def header_title(self) -> str:
        return self.display_name

    @property
    def requires_gpu(self) -> bool:
        return False

    @property
    def auth_required(self) -> bool:
        return False

    @property
    def processing_params(self) -> List[Dict[str, Any]]:
        return []

    @property
    def supports_demucs_preprocessing(self) -> bool:
        return False

    @property
    def supports_bandwidth_extension(self) -> bool:
        return False

    @property
    def supports_chunked_processing(self) -> bool:
        return False

    @abstractmethod
    def is_available(self) -> bool:
        ...

    @abstractmethod
    def load(self, progress_cb: Optional[Callable[[str], None]] = None) -> None:
        ...

    @abstractmethod
    def unload(self) -> None:
        ...

    @abstractmethod
    def process(self, request: EnhancementRequest,
                progress_cb: Optional[Callable[[str], None]] = None) -> EnhancementResult:
        ...

    @abstractmethod
    def download(self, model_dir: Path,
                 progress_cb: Optional[Callable[[str], None]] = None) -> None:
        ...


_BACKEND_REGISTRY: Dict[str, Type[AudioEnhancerBackend]] = {}


def register_backend(cls: Type[AudioEnhancerBackend]) -> Type[AudioEnhancerBackend]:
    instance = cls()
    _BACKEND_REGISTRY[instance.model_id] = cls
    return cls


def all_backends() -> List[Type[AudioEnhancerBackend]]:
    return list(_BACKEND_REGISTRY.values())


def get_backend(model_id: str) -> Type[AudioEnhancerBackend]:
    return _BACKEND_REGISTRY[model_id]


def create_backend(model_id: str) -> AudioEnhancerBackend:
    return _BACKEND_REGISTRY[model_id]()


def _active_venv_folder_name() -> Optional[str]:
    venv = os.environ.get("VIRTUAL_ENV")
    if venv:
        return Path(venv).name.lower()

    conda_env = os.environ.get("CONDA_DEFAULT_ENV")
    if conda_env:
        return Path(conda_env).name.lower()

    if sys.prefix != sys.base_prefix:
        return Path(sys.prefix).name.lower()

    return None


def detect_active_backends() -> List[Type[AudioEnhancerBackend]]:
    folder_name = _active_venv_folder_name()
    matched: List[Type[AudioEnhancerBackend]] = []

    if folder_name:
        for cls in all_backends():
            instance = cls()
            names = [n.lower() for n in instance.venv_names]
            if folder_name in names:
                matched.append(cls)

    if matched:
        return matched

    return all_backends()
