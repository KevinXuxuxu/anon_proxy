from pathlib import Path

from anon_proxy.eval import LabeledExample, Span, compute_metrics, canonical_label, span_match


def test_canonical_label_normalizes_ml_and_regex_and_opf():
    assert canonical_label("private_email") == "EMAIL"
    assert canonical_label("email") == "EMAIL"
    assert canonical_label("EMAIL") == "EMAIL"
    assert canonical_label("private_phone") == "PHONE"
    assert canonical_label("phone") == "PHONE"
    assert canonical_label("PHONE_NANP") == "PHONE"
    assert canonical_label("PHONE_INTL") == "PHONE"
    assert canonical_label("private_person") == "PERSON"
    assert canonical_label("person") == "PERSON"
    assert canonical_label("IPV4") == "IP"
    assert canonical_label("IPV6") == "IP"
    assert canonical_label("UNKNOWN_LABEL") == "UNKNOWN_LABEL"


def test_span_match_requires_exact_offsets_and_canonical_label():
    p = Span(label="private_email", start=0, end=5)
    t = Span(label="EMAIL", start=0, end=5)
    assert span_match(p, t)


def test_span_match_offsets_must_match():
    p = Span(label="EMAIL", start=0, end=4)
    t = Span(label="EMAIL", start=0, end=5)
    assert not span_match(p, t)


def test_span_match_different_canonical_labels_no_match():
    p = Span(label="EMAIL", start=0, end=5)
    t = Span(label="PHONE", start=0, end=5)
    assert not span_match(p, t)


def test_compute_metrics_perfect_recall():
    examples = [
        LabeledExample(
            text="hi",
            truth=[Span(label="EMAIL", start=0, end=2)],
            predictions=[Span(label="EMAIL", start=0, end=2)],
        ),
    ]
    m = compute_metrics(examples)
    assert m["EMAIL"]["precision"] == 1.0
    assert m["EMAIL"]["recall"] == 1.0
    assert m["EMAIL"]["f1"] == 1.0
    assert m["EMAIL"]["n"] == 1


def test_compute_metrics_half_recall():
    examples = [
        LabeledExample(
            text="hi",
            truth=[Span(label="EMAIL", start=0, end=2)],
            predictions=[Span(label="EMAIL", start=0, end=2)],
        ),
        LabeledExample(
            text="hi",
            truth=[Span(label="EMAIL", start=0, end=2)],
            predictions=[],
        ),
    ]
    m = compute_metrics(examples)
    assert m["EMAIL"]["precision"] == 1.0
    assert m["EMAIL"]["recall"] == 0.5
    assert abs(m["EMAIL"]["f1"] - (2 * 1.0 * 0.5 / 1.5)) < 1e-9


def test_compute_metrics_false_positive_drops_precision():
    examples = [
        LabeledExample(
            text="hi",
            truth=[],
            predictions=[Span(label="EMAIL", start=0, end=2)],
        ),
    ]
    m = compute_metrics(examples)
    assert m["EMAIL"]["precision"] == 0.0
    assert m["EMAIL"]["recall"] == 0.0  # no truths → recall conventionally 0


def test_compute_metrics_canonicalizes_truth_and_pred():
    """OPF uses 'email'; baseline emits 'EMAIL'; both should canonicalize to EMAIL."""
    examples = [
        LabeledExample(
            text="hi",
            truth=[Span(label="email", start=0, end=2)],
            predictions=[Span(label="EMAIL", start=0, end=2)],
        ),
    ]
    m = compute_metrics(examples)
    assert m["EMAIL"]["precision"] == 1.0
    assert m["EMAIL"]["recall"] == 1.0
