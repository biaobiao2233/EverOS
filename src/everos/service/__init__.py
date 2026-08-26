"""Application layer.

Orchestrates memory-layer capabilities into complete use cases. One CLI
command or API endpoint maps to one service method.

External usage:
    from everos.service import MemorizeResult, get, memorize, search
"""

from ._deferred_errors import MemoryMessageConflictError as MemoryMessageConflictError
from ._deferred_errors import MemoryMessageRecoveryError as MemoryMessageRecoveryError
from .get import get as get
from .memorize import BackgroundFlushResult as BackgroundFlushResult
from .memorize import MemorizeResult as MemorizeResult
from .memorize import MemoryOperationConflictError as MemoryOperationConflictError
from .memorize import MemoryOperationRecoveryError as MemoryOperationRecoveryError
from .memorize import MemoryOperationStatus as MemoryOperationStatus
from .memorize import PublishResult as PublishResult
from .memorize import StageResult as StageResult
from .memorize import get_memory_operation_status as get_memory_operation_status
from .memorize import memorize as memorize
from .memorize import publish as publish
from .memorize import queue_background_flush as queue_background_flush
from .memorize import stage as stage
from .search import search as search

__all__ = [
    "BackgroundFlushResult",
    "MemoryMessageConflictError",
    "MemoryMessageRecoveryError",
    "MemoryOperationConflictError",
    "MemoryOperationRecoveryError",
    "MemoryOperationStatus",
    "MemorizeResult",
    "PublishResult",
    "StageResult",
    "get",
    "get_memory_operation_status",
    "memorize",
    "publish",
    "queue_background_flush",
    "search",
    "stage",
]
