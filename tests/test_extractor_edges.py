"""
Extraction edges: the model's answer is brought into the schema's shape, and
nothing the transcript does not contain survives as a quote.

`_coerce` is the whole of "what the model said" becoming "what the schema
promised", so every field type is fed every wrong shape and the output pinned.
The helpers that feed the extraction prompt -- JSON extraction from a reply,
redaction, glossary corrections, consent detection, the confidence warning --
have their remaining edges pinned alongside for the same reason: each one
decides what the model sees or what a person is told, and a silent default is
the failure mode.
"""

from __future__ import annotations

import logging

import pytest

from _fixtures import build_sandbox
from plaud_bridge.asr.confidence import SUSPECT, UNRELIABLE, Assessment, prompt_warning
from plaud_bridge.compliance.consent import detect_consent
from plaud_bridge.compliance.redact import RedactionReport, redact_text
from plaud_bridge.config import Glossary
from plaud_bridge.correct.glossary import CorrectionReport, apply_corrections
from plaud_bridge.llm.base import LLMError, LLMResponse, extract_json
from plaud_bridge.models import Segment, Transcript
from plaud_bridge.profiles.extractor import (
    MAX_EXTRACTION_CHARS,
    _coerce,
    _max_tokens,
    _quote_texts,
    _type_default,
    _verify_quotes,
    extract,
    quote_is_present,
)


class _Cfg:
    """Just a dotted-key store, for the helpers that only read one setting."""

    def __init__(self, **values):
        self.values = values

    def get(self, key, default=None):
        return self.values.get(key, default)


# =========================================================================
# Type defaults and coercion
# =========================================================================
@pytest.mark.parametrize("type_name,expected", [
    ("list[quote]", []), ("List[string]", []), ("boolean", False),
    ("int", 0), ("integer", 0), ("float", 0.0), ("number", 0.0),
    ("string", ""), ("object", ""),
])
def test_each_field_type_has_the_empty_value_its_shape_implies(type_name, expected):
    assert _type_default(type_name) == expected
    assert _coerce(None, type_name) == expected, "None must become the type's empty value"


@pytest.mark.parametrize("value,expected", [
    (["a", "b"], ["a", "b"]),
    ("one item", ["one item"]),
    ({"text": "a quote"}, [{"text": "a quote"}]),
    ("", []),
    ({}, []),
    (42, []),
    (True, []),
])
def test_a_list_field_wraps_a_lone_value_and_discards_the_unlistable(value, expected):
    assert _coerce(value, "list[quote]") == expected


@pytest.mark.parametrize("value,expected", [
    (True, True), (False, False),
    ("true", True), (" YES ", True), ("1", True),
    ("false", False), ("no", False), ("", False), (0, False), ("maybe", False),
])
def test_a_boolean_field_reads_the_usual_spellings_and_nothing_else(value, expected):
    assert _coerce(value, "boolean") is expected


@pytest.mark.parametrize("value,expected", [("12", 12), (7.9, 7), ("twelve", 0), ([1], 0)])
def test_an_int_field_parses_or_falls_to_zero(value, expected):
    assert _coerce(value, "integer") == expected


@pytest.mark.parametrize("value,expected", [("1.5", 1.5), (2, 2.0), ("lots", 0.0), ({}, 0.0)])
def test_a_float_field_parses_or_falls_to_zero(value, expected):
    assert _coerce(value, "number") == expected


def test_a_string_field_serialises_structures_rather_than_printing_python_repr():
    assert _coerce({"what": "sign it", "when": "tonight"}, "string") == '{"what": "sign it", "when": "tonight"}'
    assert _coerce(["a", "b"], "string") == '["a", "b"]'
    assert _coerce("café", "string") == "café"
    assert _coerce(3, "string") == "3"


# =========================================================================
# Token ceiling
# =========================================================================
def test_the_output_ceiling_defaults_when_no_provider_is_configured():
    assert _max_tokens(_Cfg()) == 8000
    assert _max_tokens(_Cfg(**{"llm.providers": []})) == 8000
    assert _max_tokens(_Cfg(**{"llm.providers": ["x"], "llm.x.max_tokens": 1234})) == 1234


# =========================================================================
# Quotes
# =========================================================================
def test_a_quote_of_only_punctuation_is_absent_rather_than_matching_everywhere():
    haystack = " nothing to see here "
    assert not quote_is_present("...", haystack)
    assert not quote_is_present("  ", haystack)
    assert quote_is_present("to see", haystack)


def test_quote_texts_reads_plain_strings_as_well_as_quote_objects():
    assert _quote_texts("he said this") == ["he said this"]
    assert _quote_texts(["a", {"text": "b"}, {"speaker": "x"}, "", None, {"text": " "}]) == ["a", "b"]


def test_a_plain_string_quote_that_is_not_in_the_transcript_is_dropped(tmp_path, monkeypatch):
    """The schema says quote; a string that was never said is still a fabrication."""
    cfg, _ = build_sandbox(tmp_path, monkeypatch)
    profile = cfg.profile("insurance_agent")
    fields = {"stated_needs": ["We just had our second kid", "We want a yacht"],
              "meeting_type": "fact_find"}
    cleaned, dropped = _verify_quotes(fields, profile, "Marcus: We just had our second kid.")
    assert cleaned["stated_needs"] == ["We just had our second kid"]
    assert dropped == ["We want a yacht"]
    assert cleaned["meeting_type"] == "fact_find", "a non-quote field is left alone"


# =========================================================================
# extract(): the empty and the oversized transcript
# =========================================================================
def test_an_empty_transcript_is_an_error_result_and_no_model_is_called(tmp_path, monkeypatch):
    cfg, _ = build_sandbox(tmp_path, monkeypatch)
    called = []
    monkeypatch.setattr("plaud_bridge.profiles.extractor.complete_json",
                        lambda *a, **k: called.append(1))
    profile = cfg.profile("insurance_agent")
    result = extract(Transcript(segments=[Segment(0, 1, "  ", "A")]), profile, cfg)
    assert result.error == "empty transcript"
    assert result.fields == {spec.key: _type_default(spec.type) for spec in profile.fields}
    assert result.fields["stated_needs"] == [] and result.fields["next_action"] == ""
    assert not called


def test_a_provider_failure_is_an_error_result_with_empty_fields_not_a_raise(tmp_path, monkeypatch):
    cfg, _ = build_sandbox(tmp_path, monkeypatch)

    def down(*_a, **_k):
        raise LLMError("every provider refused (test)")

    monkeypatch.setattr("plaud_bridge.profiles.extractor.complete_json", down)
    profile = cfg.profile("insurance_agent")
    result = extract(Transcript(segments=[Segment(0, 1, "hello", "A")]), profile, cfg)
    assert result.error == "every provider refused (test)"
    assert result.fields == {spec.key: _type_default(spec.type) for spec in profile.fields}
    assert result.llm_provider == "" and result.cost_usd == 0.0
    assert not result.requires_human_attention and result.unverified_quotes == 0


def test_an_oversized_transcript_is_truncated_and_the_model_is_told(tmp_path, monkeypatch):
    cfg, _ = build_sandbox(tmp_path, monkeypatch)
    seen = {}

    def fake(cfg, system, user, local_only=False, max_tokens=None):
        seen["user"] = user
        return {}, LLMResponse(provider="stub", model="stub")

    monkeypatch.setattr("plaud_bridge.profiles.extractor.complete_json", fake)
    body = "word " * (MAX_EXTRACTION_CHARS // 5 + 500)
    extract(Transcript(), cfg.profile("unfiled"), cfg, transcript_text=body)
    assert "[... transcript truncated ...]" in seen["user"]
    start = seen["user"].index("<<<BEGIN TRANSCRIPT>>>\n") + len("<<<BEGIN TRANSCRIPT>>>\n")
    end = seen["user"].index("\n<<<END TRANSCRIPT>>>")
    assert seen["user"][start:end] == body[:MAX_EXTRACTION_CHARS] + "\n\n[... transcript truncated ...]"


# =========================================================================
# extract_json: fences and empties
# =========================================================================
def test_json_inside_a_code_fence_is_recovered_when_the_reply_has_commentary():
    reply = 'Sure, here you go:\n```json\n{"a": 1}\n```\nLet me know if you need more.'
    assert extract_json(reply) == {"a": 1}
    # A bare fence with no language tag, and a brace-only tail, both work.
    assert extract_json("```\n{\"b\": [2]}\n```") == {"b": [2]}
    assert extract_json("prose {\"c\": true} trailing") == {"c": True}


def test_a_blank_or_non_object_reply_is_an_llm_error_carrying_the_reply():
    with pytest.raises(LLMError, match="did not return parseable JSON"):
        extract_json("   ")
    with pytest.raises(LLMError) as excinfo:
        extract_json("[1, 2, 3]")
    assert "[1, 2, 3]" in str(excinfo.value), "the reply head must be in the error for debugging"


# =========================================================================
# Redaction
# =========================================================================
def test_an_empty_redaction_report_says_so():
    assert RedactionReport().summary() == "no redactions"
    assert RedactionReport().total == 0


def test_an_invalid_pattern_is_skipped_by_name_and_the_rest_still_redact(caplog):
    patterns = {"broken": "(", "ssn": r"\b\d{3}-\d{2}-\d{4}\b"}
    with caplog.at_level(logging.WARNING, logger="plaud_bridge.redact"):
        out, report = redact_text("ssn 123-45-6789 and (parens)", patterns)
    assert out == "ssn [SSN_REDACTED] and (parens)"
    assert report.counts == {"ssn": 1}
    assert any("invalid redaction pattern 'broken'" in r.getMessage() for r in caplog.records)


def test_redacting_empty_text_is_a_no_op():
    out, report = redact_text("", {"ssn": r"\d"})
    assert out == "" and report.total == 0


# =========================================================================
# Glossary corrections
# =========================================================================
def test_an_empty_correction_report_says_so():
    assert CorrectionReport().summary() == "no corrections applied"


def test_a_missing_or_empty_glossary_leaves_segments_untouched():
    segs = [Segment(0, 1, "Para med exam.", "A")]
    out, report = apply_corrections(segs, None)
    assert out is segs and report.total == 0
    out, report = apply_corrections(segs, Glossary())
    assert out[0].text == "Para med exam." and report.total == 0


def test_a_sentence_initial_correction_keeps_its_capital_letter():
    glossary = Glossary(corrections={"para med": "paramed", "i u l": "IUL"})
    segs = [Segment(0, 1, "Para med exam next week.", "A"),
            Segment(1, 2, "the i u l policy", "A")]
    out, report = apply_corrections(segs, glossary)
    assert out[0].text == "Paramed exam next week."
    assert out[1].text == "the IUL policy", "an acronym replacement is used exactly as authored"
    assert report.applied == {"para med": 1, "i u l": 1}
    assert report.segments_touched == 2


# =========================================================================
# Consent: an empty window
# =========================================================================
def test_a_transcript_with_nothing_in_the_consent_window_says_so():
    tr = Transcript(segments=[Segment(200, 203, "I record these calls, okay?", "Sasson"),
                              Segment(203, 206, "Sure.", "Client")])
    result = detect_consent(tr, window_seconds=90)
    assert not result.announced and not result.agreed and not result.complete
    assert result.notes == ["transcript has no content in the consent window"]
    assert detect_consent(Transcript()).notes == ["transcript has no content in the consent window"]


# =========================================================================
# Confidence: what the extraction prompt is told
# =========================================================================
def test_a_suspect_transcript_gets_a_softer_warning_than_an_unreliable_one():
    suspect = prompt_warning(Assessment(verdict=SUSPECT))
    unreliable = prompt_warning(Assessment(verdict=UNRELIABLE))
    assert suspect.startswith("TRANSCRIPT RELIABILITY: parts of this transcript")
    assert "leave it out rather than interpreting it" in suspect
    assert unreliable != suspect and "Prefer empty fields" in unreliable
    assert prompt_warning(Assessment()) == ""
