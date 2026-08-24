"""emullm: the simulated LLM backend relay (standalone).

Relays OpenAI-compatible HTTP requests over a WebSocket to a connected
worker (human or agent) acting as the model. See docs/EMULLM_RELAY.md
for the design and docs/EMULLM_ONBOARD.md for the worker how-to.
"""
from __future__ import annotations

from .api import register_doc_alias, register_mock_workers, unregister_mock_workers

__all__ = ["register_doc_alias", "register_mock_workers", "unregister_mock_workers"]
