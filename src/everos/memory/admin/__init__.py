"""Read-only administration domain for local EverOS operators."""

from .dto import (
    AdminMemoryFileContentResponse,
    AdminMemoryFilesResponse,
    AdminMemoryKind,
    PipelineStatusResponse,
)
from .manager import (
    AdminFileNotFoundError,
    AdminFileTooLargeError,
    AdminInvalidPathError,
    get_memory_file_content,
    get_pipeline_status,
    list_memory_files,
)

__all__ = [
    "AdminFileNotFoundError",
    "AdminFileTooLargeError",
    "AdminInvalidPathError",
    "AdminMemoryFileContentResponse",
    "AdminMemoryFilesResponse",
    "AdminMemoryKind",
    "PipelineStatusResponse",
    "get_memory_file_content",
    "get_pipeline_status",
    "list_memory_files",
]
