import pytest

from anon_proxy.signatures import compute_signature, NoLeakageError


# ---------- PHONE ----------

def test_phone_signature_us_format():
    sig = compute_signature("PHONE", "(555) 867-5309")
    assert sig == {"label": "PHONE", "country_code": None, "digit_count": 10, "separator_pattern": "() -"}


def test_phone_signature_uk_e164():
    sig = compute_signature("PHONE", "+44 7700 900123")
    assert sig == {"label": "PHONE", "country_code": "+44", "digit_count": 10, "separator_pattern": "+ "}


def test_phone_signature_no_leakage():
    """Critical: signature output must not contain >2 raw chars from digit input."""
    raw_digits = "5558675309"
    sig = compute_signature("PHONE", "555-867-5309")
    serialized = str(sig)
    for i in range(len(raw_digits) - 2):
        chunk = raw_digits[i : i + 3]
        assert chunk not in serialized, f"raw chunk {chunk!r} leaked into signature {serialized!r}"


# ---------- EMAIL ----------

def test_email_signature_classifies_domain():
    sig = compute_signature("EMAIL", "alice@gmail.com")
    assert sig == {"label": "EMAIL", "domain_class": "free", "local_len": 5, "tld": "com"}


def test_email_signature_enterprise():
    sig = compute_signature("EMAIL", "j.smith@acme-corp.io")
    assert sig["domain_class"] == "enterprise"
    assert sig["tld"] == "io"


def test_email_signature_edu():
    sig = compute_signature("EMAIL", "student@mit.edu")
    assert sig["domain_class"] == "edu"


def test_email_signature_no_leakage():
    """Local-part 'very-unique-name-1234' must not appear in signature."""
    sig = compute_signature("EMAIL", "very-unique-name-1234@gmail.com")
    sig_str = str(sig)
    assert "very" not in sig_str
    assert "unique" not in sig_str
    assert "1234" not in sig_str


# ---------- PERSON ----------

def test_person_signature_first_last():
    sig = compute_signature("PERSON", "Alice Smith")
    assert sig == {
        "label": "PERSON", "token_count": 2, "caps_pattern": "UU",
        "has_hyphen": False, "has_apostrophe": False, "total_len": 11,
    }


def test_person_signature_hyphenated():
    sig = compute_signature("PERSON", "Anne-Marie O'Brien")
    assert sig["token_count"] == 2
    assert sig["has_hyphen"] is True
    assert sig["has_apostrophe"] is True


def test_person_signature_no_leakage():
    """Names must not appear in signature output."""
    sig = compute_signature("PERSON", "Boyu Liu")
    sig_str = str(sig)
    assert "Boyu" not in sig_str
    assert "Liu" not in sig_str


# ---------- SSN ----------

def test_ssn_signature():
    sig = compute_signature("SSN", "123-45-6789")
    assert sig == {"label": "SSN", "format": "NNN-NN-NNNN"}


def test_ssn_signature_no_leakage():
    sig = compute_signature("SSN", "987-65-4321")
    sig_str = str(sig)
    # No 3-digit substring from the input should appear
    assert "987" not in sig_str
    assert "654" not in sig_str
    assert "321" not in sig_str


# ---------- IP ----------

def test_ip_signature_v4():
    sig = compute_signature("IP", "192.168.0.1")
    assert sig == {"label": "IP", "format": "v4"}


def test_ip_signature_v6():
    sig = compute_signature("IP", "2001:db8::1")
    assert sig["format"] == "v6"


def test_ip_signature_no_leakage():
    sig = compute_signature("IP", "203.0.113.42")
    sig_str = str(sig)
    assert "203" not in sig_str
    assert "113" not in sig_str
    assert "42" not in sig_str


# ---------- CC ----------

def test_cc_signature_visa():
    sig = compute_signature("CC", "4111 1111 1111 1111")
    assert sig == {"label": "CC", "brand": "visa", "digit_count": 16}


def test_cc_signature_mastercard():
    sig = compute_signature("CC", "5500 0000 0000 0004")
    assert sig["brand"] == "mastercard"


def test_cc_signature_amex():
    sig = compute_signature("CC", "3782 822463 10005")
    assert sig["brand"] == "amex"
    assert sig["digit_count"] == 15


# ---------- ADDRESS ----------

def test_address_signature_us():
    sig = compute_signature("ADDRESS", "1600 Pennsylvania Ave NW, Washington, DC 20500")
    assert sig["has_zip"] is True
    assert sig["line_count"] == 1


def test_address_signature_with_unit():
    sig = compute_signature("ADDRESS", "123 Main St Apt 4B, Springfield, IL 62701")
    assert sig["has_unit"] is True
    assert sig["has_zip"] is True


def test_address_signature_no_leakage():
    sig = compute_signature("ADDRESS", "1600 Pennsylvania Ave NW, Washington, DC 20500")
    sig_str = str(sig)
    assert "1600" not in sig_str
    assert "Pennsylvania" not in sig_str
    assert "Washington" not in sig_str
    assert "20500" not in sig_str


# ---------- Unknown label ----------

def test_unknown_label_returns_length_only():
    sig = compute_signature("UNKNOWN_LABEL", "some random text 12345")
    assert sig == {"label": "UNKNOWN_LABEL", "shape": "unknown", "len": 22}
