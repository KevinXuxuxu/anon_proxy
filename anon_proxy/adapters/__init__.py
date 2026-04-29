"""API provider adapters for request/response masking.

Each adapter implements:
- mask_request(body, masker) -> masked_body
- unmask_response(body, masker) -> unmasked_body
- transform_stream(upstream_bytes, masker) -> async iterator of bytes

Available adapters:
- anthropic: Anthropic Messages API
- openai: OpenAI Chat Completions API
"""

from anon_proxy.adapters import anthropic
from anon_proxy.adapters import openai

__all__ = ["anthropic", "openai"]
