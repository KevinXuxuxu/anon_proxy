"""Per-label structural signatures: turn raw entity text into LLM-safe abstractions.

Signatures are designed so that the LLM (via the triage skill) can reason about
the SHAPE of detected/missed entities without ever seeing the raw values. Each
new label MUST pass the corresponding leakage test in tests/test_signatures.py
before being merged.

Audit (manual review of leakage prevention):
- PHONE: leaks separators (e.g., "() -") and country code prefix; digit count is structural.
- EMAIL: leaks tld (e.g., "com") and domain class; local-part length is structural.
- PERSON: leaks token count, caps pattern, hyphen/apostrophe presence; never names.
- ADDRESS: leaks has-zip, has-unit, line count, total length; never literal address.
- SSN: leaks the format string ("NNN-NN-NNNN"); never the digits.
- IP: leaks "v4" / "v6"; never the address.
- CC: leaks brand + digit count; never the digits.

The deliberately-leaked features (separators, TLDs, format) are *low entropy*
and would not let an attacker reconstruct the original PII. They are useful
for regex synthesis (the skill needs to know "+44 prefix" to write a UK regex).
"""

from __future__ import annotations

import re


class NoLeakageError(Exception):
    """Raised by tests when a signature accidentally leaks raw substrings."""


def compute_signature(label: str, text: str) -> dict:
    """Return a structural signature for `text` under `label`.

    The signature is LLM-safe: no raw substrings from `text` appear in the output
    (modulo deliberately-low-entropy structural features documented per-label).

    Unknown labels fall back to a length-only signature.
    """
    impl = _IMPL.get(label.upper())
    if impl is None:
        return {"label": label, "shape": "unknown", "len": len(text)}
    return {"label": label, **impl(text)}


# ---------- per-label implementations ----------


def _phone(text: str) -> dict:
    m = re.match(r"^(\+\d{1,3})([\s\-]?)", text)
    if m and m.group(1):
        cc = m.group(1)
        # Strip country code and its trailing separator from the remainder.
        remainder = text[m.end():]
        # separator_pattern = the "+" literal plus the boundary separator char(s).
        # Internal digit-group separators within the national number are not included —
        # they are formatting noise, not structurally meaningful for regex synthesis.
        separator_prefix = "+" + m.group(2)
    else:
        cc = None
        remainder = text
        separator_prefix = ""
    digits = re.sub(r"\D", "", remainder)
    # For no-CC numbers, drop digits and letters to expose formatting chars.
    # For CC numbers, only the boundary separator matters (see above).
    if cc is None:
        separators = re.sub(r"[\dA-Za-z]", "", remainder)
    else:
        separators = separator_prefix
    return {"country_code": cc, "digit_count": len(digits), "separator_pattern": separators}


_FREE_DOMAINS = {
    "gmail.com", "yahoo.com", "hotmail.com", "outlook.com",
    "icloud.com", "proton.me", "protonmail.com",
}


def _email(text: str) -> dict:
    if "@" not in text:
        return {"domain_class": "invalid", "local_len": len(text), "tld": None}
    local, domain = text.rsplit("@", 1)
    domain_lower = domain.lower()
    tld = domain_lower.rsplit(".", 1)[-1] if "." in domain_lower else None
    if domain_lower in _FREE_DOMAINS:
        cls = "free"
    elif tld == "edu":
        cls = "edu"
    elif tld == "gov":
        cls = "gov"
    elif tld == "mil":
        cls = "mil"
    else:
        cls = "enterprise"
    return {"domain_class": cls, "local_len": len(local), "tld": tld}


def _person(text: str) -> dict:
    tokens = text.split()
    caps = "".join("U" if t and t[0].isupper() else "l" for t in tokens)
    return {
        "token_count": len(tokens),
        "caps_pattern": caps,
        "has_hyphen": "-" in text,
        "has_apostrophe": "'" in text or "’" in text,
        "total_len": len(text),
    }


def _ssn(text: str) -> dict:
    fmt = re.sub(r"\d", "N", text)
    return {"format": fmt}


def _ip(text: str) -> dict:
    return {"format": "v6" if ":" in text else "v4"}


def _cc(text: str) -> dict:
    digits = re.sub(r"\D", "", text)
    brand = "unknown"
    if digits.startswith("4"):
        brand = "visa"
    elif digits[:2] in {"51", "52", "53", "54", "55"}:
        brand = "mastercard"
    elif digits[:2] in {"34", "37"}:
        brand = "amex"
    return {"brand": brand, "digit_count": len(digits)}


def _address(text: str) -> dict:
    return {
        "has_zip": bool(re.search(r"\b\d{5}(?:-\d{4})?\b", text)),
        "has_unit": bool(re.search(r"\b(apt|suite|unit|#)\b", text, re.I)),
        "line_count": text.count("\n") + 1,
        "total_len": len(text),
    }


_IMPL = {
    "PHONE": _phone,
    "EMAIL": _email,
    "PERSON": _person,
    "SSN": _ssn,
    "IP": _ip,
    "CC": _cc,
    "ADDRESS": _address,
}
