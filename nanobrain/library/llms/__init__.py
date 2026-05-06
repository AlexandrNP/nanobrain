# Nanobrain Local LLM Support
# OpenAI-interchangeable local LLM clients with shared server support

from .local_llm_client import LocalLLM
from .shared_vllm_server import SharedvLLMServer

# Backward compatibility (deprecated)
from .vllm_server_manager import vLLMServerManager

__all__ = [
    'LocalLLM',
    'SharedvLLMServer',
    'vLLMServerManager',  # Deprecated - use SharedvLLMServer
]
