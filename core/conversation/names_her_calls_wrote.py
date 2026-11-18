"""Whether a dotted name in her reply is one a call of hers wrote this turn."""
from __future__ import annotations


def a_name_her_calls_wrote(token: str) -> bool:
    """LIVE 18 November 2026 her account of what she ran, "I ran sympy.isprime", was marked as two sentences run
    together. What she wrote in a call this turn is a name, whatever it looks like."""
    from core.conversation.surface_disposition import turn_tool_receipts

    return bool(token) and any(token in str(receipt.get("as_made") or "") for receipt in turn_tool_receipts())
