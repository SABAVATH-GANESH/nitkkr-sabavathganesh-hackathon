import pytest

from src.nlp_engine.model import (KeywordEventClassifier, LexiconSentiment, RiskEngine, compute_impact,
                                  source_credibility)
from src.schemas.risk_signal import EventType, RawArticle


def art(i, title, source="Reuters"):
    return RawArticle(article_id=str(i), published_at="2026-01-01T00:00:00Z", source=source, title=title)


def test_lexicon_sign_negation_and_empty():
    lx = LexiconSentiment()
    assert lx.score("Company beats estimates, record profit")[0] > 0.4
    assert lx.score("Shares plunge after fraud probe")[0] < -0.4
    assert lx.score("not a loss")[0] > 0           # negated negative
    assert lx.score("Annual meeting scheduled") == (0.0, 0.3)
    assert -1 < lx.score("beat " * 50)[0] < 1


def test_event_classifier_word_boundaries_and_neutral():
    ev = KeywordEventClassifier()
    assert ev.classify("ICICIBANK to hold annual general meeting")[0] == EventType.NEUTRAL  # 'ban' inside 'bank'
    assert ev.classify("Quarterly earnings and revenue beat")[0] == EventType.EARNINGS
    assert ev.classify("Ransomware breach hits bank")[0] == EventType.CYBER_OPS
    assert 0 < ev.classify("SEBI probe and fine")[1] <= 1


def test_new_event_classes_and_stems():
    ev = KeywordEventClassifier()
    assert ev.classify("Firm credit rating downgraded after missed payment")[0] == EventType.CREDIT_EVENT
    assert ev.classify("Company launches new product, unveils platform")[0] == EventType.PRODUCT_LAUNCH
    assert ev.classify("War, sanctions and border tensions")[0] == EventType.GEOPOLITICAL
    assert ev.classify("RBI rate hike as inflation and GDP slump")[0] == EventType.MACRO


def test_strong_geopolitical_news_exceeds_trigger_threshold():
    s = RiskEngine(use_transformers=False).process(
        [art(1, "War escalates as sanctions and border tensions spark market crash fears", "Reuters")])[0]
    assert s.event_type == EventType.GEOPOLITICAL and s.impact > 7 and s.sentiment < -0.5
    t = RiskEngine(use_transformers=False).process([art(2, "war sanctions tensions crash lol", "Twitter")])[0]
    assert t.impact < s.impact          # lower-credibility source scores lower


def test_impact_bounds_monotonic_and_credibility():
    lo = compute_impact(0, 0.3, EventType.NEUTRAL, 0.5)
    hi = compute_impact(-1, 1, EventType.GEOPOLITICAL, 1, 1.0)
    assert 1.0 <= lo < 2.0 and 1.0 <= hi <= 10.0 and hi > 8
    assert compute_impact(-0.9, .9, EventType.LEGAL, .9, 1.0) > compute_impact(-0.9, .9, EventType.LEGAL, .9, 0.3)
    assert compute_impact(-0.9, .9, EventType.LEGAL, .9) == compute_impact(0.9, .9, EventType.LEGAL, .9)  # |s|
    assert compute_impact(5, 5, EventType.GEOPOLITICAL, 5, 5) == 10.0  # clipped
    assert source_credibility(" Reuters ") == 1.0 and source_credibility("unknown") == 0.7


def test_engine_batching_equivalence_and_determinism():
    arts = [art(i, t) for i, t in enumerate(["Firm beats earnings, record profit", "Hack breach shutdown", "AGM next month"] * 5)]
    a = RiskEngine(use_transformers=False, batch_size=1).process(arts)
    b = RiskEngine(use_transformers=False, batch_size=7).process(arts)
    assert a == b and len(a) == 15
    assert all(-1 <= s.sentiment <= 1 and 1 <= s.impact <= 10 for s in a)
    assert RiskEngine(use_transformers=False).process([]) == []
    with pytest.raises(ValueError):
        RiskEngine(batch_size=0)


def test_engine_offline_fallback_without_weights():
    eng = RiskEngine()  # transformers/weights absent in CI -> must not raise
    assert eng.process([art(1, "Bank fined after probe")])[0].sentiment < 0
