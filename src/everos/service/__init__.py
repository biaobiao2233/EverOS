"""Application layer.

Orchestrates memory-layer capabilities into complete use cases. One CLI
command or API endpoint maps to one service method.

External usage:
    from everos.service import MemorizeResult, get, memorize, search
"""

from .get import get as get
from .memorize import MemorizeResult as MemorizeResult
from .memorize import MemoryOperationConflictError as MemoryOperationConflictError
from .memorize import MemoryOperationRecoveryError as MemoryOperationRecoveryError
from .memorize import MemoryOperationStatus as MemoryOperationStatus
from .memorize import get_memory_operation_status as get_memory_operation_status
from .memorize import memorize as memorize
from .search import search as search

__all__ = [
    "MemoryOperationConflictError",
    "MemoryOperationRecoveryError",
    "MemoryOperationStatus",
    "MemorizeResult",
    "get",
    "get_memory_operation_status",
    "memorize",
    "search",
]
