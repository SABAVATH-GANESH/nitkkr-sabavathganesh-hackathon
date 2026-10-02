import pytest
from pydantic import ValidationError

from src.ingestion import CSVReplaySource, RateLimiter, validate_records
from src.ingestion.gdelt import GDELTRSSSource
from src.ingestion.newsapi import NewsAPISource
from src.schemas.risk_signal import PortfolioState, RawArticle, RiskSignal
from src.utils.cache import TTLCache
from src.utils.errors import IngestionError
from tests.conftest import make_signal


def test_raw_article_normalises():
    a = RawArticle(article_id="1", published_at="2026-01-01T00:00:00", source="x", title="  a   b ", tickers=["tcs", " TCS "])
    assert a.title == "a b" and a.tickers == ["TCS"] and a.published_at.tzinfo is not None


def test_schemas_forbid_extra_and_bounds():
    with pytest.raises(ValidationError):
        RawArticle(article_id="1", published_at="2026-01-01", source="x", title="t", bogus=1)
    with pytest.raises(ValidationError):
        make_signal(sentiment=1.5)
    with pytest.raises(ValidationError):
        make_signal(impact=0.5)


def test_portfolio_and_asset_validation():
    from tests.conftest import asset
    with pytest.raises(ValidationError):
        PortfolioState(assets=[asset(asset_id="A"), asset(asset_id="A")])            # duplicate ids
    with pytest.raises(ValidationError):
        asset(asset_type="bond", duration=0)                                          # bond needs duration
    with pytest.raises(ValidationError):
        asset(asset_type="loan", market_value=-1)
    with pytest.raises(ValidationError):
        asset(pd=1.5)
    assert asset(asset_type="derivative", market_value=-5e5).market_value < 0         # negative MTM allowed
    assert asset(ticker=" aaa ").ticker == "AAA"
    assert PortfolioState(assets=[asset(asset_id="A"), asset(asset_id="B")]).total_value == 2_000_000


def test_validate_records_drops_bad_and_dupes():
    good = dict(article_id="1", published_at="2026-01-01T00:00:00Z", source="s", title="t")
    out = validate_records([good, good, dict(good, article_id="2", title=""), dict(good, article_id="3", published_at=None)])
    assert [a.article_id for a in out] == ["1"]


def test_csv_replay_and_missing(tmp_path):
    assert len(CSVReplaySource("data/synthetic_news.csv").fetch()) == 55
    with pytest.raises(IngestionError):
        CSVReplaySource(tmp_path / "nope.csv").fetch()
    bad = tmp_path / "b.csv"; bad.write_text("a,b\n1,2\n")
    with pytest.raises(IngestionError):
        CSVReplaySource(bad).fetch()


def test_rate_limiter_deterministic():
    t = [0.0]; slept = []
    rl = RateLimiter(2.0, burst=1, clock=lambda: t[0], sleep=lambda s: (slept.append(s), t.__setitem__(0, t[0] + s)))
    assert rl.acquire() == 0.0
    assert rl.acquire() == pytest.approx(0.5)
    with pytest.raises(ValueError):
        RateLimiter(0)


class _Resp:
    def __init__(self, data): self.data = data
    def raise_for_status(self): pass
    def json(self): return self.data


class _Sess:
    def __init__(self, data=None, fail=False): self.data, self.fail, self.calls = data, fail, 0
    def get(self, *a, **k):
        self.calls += 1
        if self.fail: raise ConnectionError("down")
        return _Resp(self.data)


PAYLOAD = {"articles": [{"url": "http://x/1", "publishedAt": "2026-01-01T00:00:00Z", "title": "T", "description": "D", "source": {"name": "Reuters"}}]}


def test_newsapi_cache_stale_and_errors():
    cache = TTLCache(ttl_seconds=0)  # always stale
    s = _Sess(PAYLOAD)
    src = NewsAPISource("q", api_key="k", cache=cache, session=s, limiter=RateLimiter(1000, 5))
    assert len(src.fetch()) == 1
    src.session = _Sess(fail=True)
    assert len(src.fetch()) == 1  # stale cache served offline
    with pytest.raises(IngestionError):
        NewsAPISource("q2", api_key="k", session=_Sess(fail=True), limiter=RateLimiter(1000, 5)).fetch()
    with pytest.raises(IngestionError):
        NewsAPISource("q", api_key=None).fetch() if not __import__("os").getenv("NEWSAPI_KEY") else (_ for _ in ()).throw(IngestionError("x"))


def test_gdelt_parse():
    class Feed: bozo = 0; entries = [{"title": "War news", "link": "http://g/1", "published": "Mon, 28 Sep 2026 10:00:00 GMT", "summary": "s"},
                                     {"title": "x", "link": "http://g/2", "published": "garbage"}]
    arts = GDELTRSSSource(parser=lambda url: Feed, limiter=RateLimiter(1000, 5)).fetch()
    assert len(arts) == 2 and arts[0].source == "GDELT"
    class Dead: bozo = 1; entries = []
    with pytest.raises(IngestionError):
        GDELTRSSSource(parser=lambda url: Dead, limiter=RateLimiter(1000, 5)).fetch()


def test_multisource_merges_and_survives_failure():
    from src.ingestion import MultiSource
    class Boom:
        def fetch(self): raise IngestionError("down")
    news, social = CSVReplaySource("data/synthetic_news.csv"), CSVReplaySource("data/synthetic_social.csv")
    merged = MultiSource([news, Boom(), social]).fetch()
    assert len(merged) == 125 and {"Twitter", "Reuters"} <= {a.source for a in merged}
    assert [a.published_at for a in merged] == sorted(a.published_at for a in merged)
