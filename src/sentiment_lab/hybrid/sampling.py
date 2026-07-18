"""Return-blind company relevance, event candidates, and story clustering."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import timedelta

from sentiment_lab.config.models import ValidationUniverseMember
from sentiment_lab.data.schemas import NewsArticle
from sentiment_lab.hybrid.schemas import HybridEventType

_TOKEN = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")
_GENERIC_TITLE = re.compile(
    r"\b(?:market (?:wrap|roundup|update)|stocks? (?:to watch|today)|"
    r"top \d+ stocks?|best stocks?|sector (?:wrap|roundup)|what investors need to know|"
    r"stock market today|premarket|closing bell)\b",
    re.IGNORECASE,
)
_LISTICLE = re.compile(r"\b(?:\d+|five|ten|three)\s+(?:best|top|stocks?|companies|reasons)\b", re.I)

_EVENT_PATTERNS: tuple[tuple[HybridEventType, tuple[str, ...]], ...] = (
    (
        HybridEventType.earnings,
        (
            r"\bearnings\b",
            r"\bquarterly results?\b",
            r"\b(?:revenue|sales|eps|earnings per share|operating margin)\b.*\b(?:beat|miss|rose|fell|reported)\b",
            r"\b(?:q[1-4]|fiscal (?:quarter|year))\b.*\bresults?\b",
        ),
    ),
    (
        HybridEventType.guidance,
        (
            r"\b(?:raises?|cuts?|lowers?|withdraws?|reaffirms?) (?:its )?(?:guidance|outlook|forecast)\b",
            r"\b(?:raises?|cuts?|lowers?|withdraws?|reaffirms?)\b.{0,30}\b(?:guidance|outlook|forecast)\b",
            r"\b(?:guidance|outlook) (?:raised|lowered|cut|withdrawn|reaffirmed)\b",
            r"\bforecasts? (?:revenue|sales|earnings|eps)\b",
        ),
    ),
    (
        HybridEventType.analyst_action,
        (
            r"\b(?:upgrades?|downgrades?|initiates?|reiterates?)\b",
            r"\bprice target\b",
            r"\b(?:buy|sell|hold|overweight|underweight) rating\b",
        ),
    ),
    (
        HybridEventType.merger_acquisition,
        (
            r"\b(?:merger|acquisition|acquires?|acquired|takeover|buyout|divestiture|divests?)\b",
            r"\b(?:deal|transaction) valued at\b",
        ),
    ),
    (
        HybridEventType.regulatory,
        (
            r"\b(?:regulator|regulatory|antitrust|doj|ftc|sec|fda|eu commission)\b",
            r"\b(?:probe|investigation|approval|clearance|consent order)\b",
        ),
    ),
    (
        HybridEventType.litigation,
        (r"\b(?:lawsuit|litigation|sues?|sued|settlement|court ruling|class action|verdict)\b",),
    ),
    (
        HybridEventType.cybersecurity,
        (r"\b(?:cyberattack|cyber attack|ransomware|data breach|hacked|security breach)\b",),
    ),
    (
        HybridEventType.fraud_accounting,
        (
            r"\b(?:fraud|accounting irregularit|restatement|misstatement|whistleblower)\w*\b",
            r"\b(?:bankruptcy|chapter 11|going concern|insolven)\w*\b",
        ),
    ),
    (
        HybridEventType.dividend,
        (r"\b(?:declares?|raises?|cuts?|suspends?) (?:a |its )?(?:quarterly )?dividend\b",),
    ),
    (
        HybridEventType.buyback,
        (r"\b(?:buyback|share repurchase|repurchase program|repurchase authorization)\b",),
    ),
    (
        HybridEventType.management,
        (
            r"\b(?:appoints?|names?|resigns?|retires?|ousts?)\b.*\b(?:ceo|cfo|chair|president|executive)\b",
            r"\b(?:ceo|cfo|chief executive|chief financial officer)\b.*\b(?:resigns?|retires?|appointed|named)\b",
        ),
    ),
    (
        HybridEventType.financing,
        (r"\b(?:debt offering|equity offering|secondary offering|convertible notes?|raises? capital|refinanc)\w*\b",),
    ),
    (
        HybridEventType.restructuring,
        (r"\b(?:restructuring|reorganization|layoffs?|job cuts?|cost-cutting|strategic review)\b",),
    ),
    (
        HybridEventType.operations,
        (
            r"\b(?:outage|disruption|shutdown|strike|recall|production halt|supply shortage|plant closure)\b",
            r"\b(?:production|deliveries|shipments)\b.*\b(?:rose|fell|halted|suspended|missed)\b",
        ),
    ),
    (
        HybridEventType.product,
        (
            r"\b(?:launches?|unveils?|introduces?|approves?|approval)\b.*\b(?:product|drug|platform|service|device|model)\b",
            r"\b(?:product launch|new drug|clinical trial|phase [123]|fda approval)\b",
        ),
    ),
    (
        HybridEventType.capital_allocation,
        (r"\b(?:capital allocation|special dividend|asset sale|investment plan|capital spending|capex)\b",),
    ),
    (
        HybridEventType.macro_exposure,
        (r"\b(?:tariff|interest rates?|inflation|currency headwind|commodity prices?|recession)\b",),
    ),
)


@dataclass(frozen=True)
class PreInferenceScore:
    score: float
    eligible: bool
    components: dict[str, float]
    exclusion_reasons: tuple[str, ...]
    event_type_candidates: tuple[HybridEventType, ...]


@dataclass(frozen=True)
class ClusteredStory:
    article_id: str
    cluster_id: str
    primary: bool


def _normalize(value: str) -> str:
    return " ".join(_TOKEN.findall(unicodedata.normalize("NFKC", value).casefold()))


def _aliases(member: ValidationUniverseMember) -> tuple[str, ...]:
    raw = [member.company_name, *member.aliases]
    legal_suffixes = re.compile(
        r"\b(?:incorporated|corporation|company|inc|corp|plc|ltd|limited|holdings?|group)\b\.?",
        re.I,
    )
    stripped = legal_suffixes.sub(" ", member.company_name)
    raw.append(stripped)
    return tuple(sorted({_normalize(item) for item in raw if len(_normalize(item)) >= 3}))


def candidate_event_types(title: str, content: str) -> tuple[HybridEventType, ...]:
    """Return deterministic context candidates; never use realized returns."""

    text = _normalize(f"{title} {content[:6000]}")
    candidates = [
        event_type
        for event_type, patterns in _EVENT_PATTERNS
        if any(re.search(pattern, text, re.I) for pattern in patterns)
    ]
    return tuple(dict.fromkeys(candidates)) or (HybridEventType.other,)


def company_relevance_score(
    article: NewsArticle,
    member: ValidationUniverseMember,
    *,
    minimum_text_characters: int = 400,
    minimum_score: float = 0.55,
    maximum_symbols: int = 5,
) -> PreInferenceScore:
    """Score target-company specificity before any model or return lookup."""

    aliases = _aliases(member)
    title = _normalize(article.title)
    body = _normalize(article.content)
    opening = " ".join(body.split()[:120])
    ticker_base = member.ticker.split(".", maxsplit=1)[0].casefold()
    headline_alias = any(alias in title for alias in aliases)
    opening_alias = any(alias in opening for alias in aliases)
    body_mentions = sum(body.count(alias) for alias in aliases)
    ticker_headline = bool(
        re.search(rf"(?:\$|\b){re.escape(ticker_base)}\b", article.title, re.I)
    ) and len(ticker_base) >= 3
    symbol_match = member.ticker in article.symbols
    event_candidates = candidate_event_types(article.title, article.content)
    event_specific = event_candidates != (HybridEventType.other,)
    low_text = len(" ".join(article.content.split())) < minimum_text_characters
    generic = bool(_GENERIC_TITLE.search(article.title) or _LISTICLE.search(article.title))
    too_many_symbols = len(article.symbols) > maximum_symbols
    primary_subject = headline_alias and (opening_alias or body_mentions >= 2)

    components = {
        "headline_company": 0.30 if headline_alias else 0.0,
        "headline_ticker": 0.10 if ticker_headline else 0.0,
        "opening_company": 0.20 if opening_alias else 0.0,
        "repeated_company_mentions": 0.10 if body_mentions >= 2 else 0.0,
        "primary_subject": 0.10 if primary_subject else 0.0,
        "specific_event": 0.15 if event_specific else 0.0,
        "adequate_text": 0.05 if not low_text else 0.0,
        "generic_penalty": -0.25 if generic else 0.0,
        "multi_ticker_penalty": -0.20 if too_many_symbols else 0.0,
        "missing_provider_symbol_penalty": -0.40 if not symbol_match else 0.0,
    }
    score = max(0.0, min(1.0, sum(components.values())))
    reasons: list[str] = []
    if not symbol_match:
        reasons.append("provider_symbol_missing")
    if not (headline_alias or ticker_headline):
        reasons.append("target_not_in_headline")
    if not opening_alias:
        reasons.append("target_not_in_opening")
    if low_text:
        reasons.append("low_information_text")
    if generic:
        reasons.append("generic_or_listicle")
    if too_many_symbols:
        reasons.append("too_many_unrelated_symbols")
    if not primary_subject:
        reasons.append("target_not_primary_subject")
    eligible = score >= minimum_score and not low_text and symbol_match and primary_subject
    return PreInferenceScore(
        score=score,
        eligible=eligible,
        components=components,
        exclusion_reasons=tuple(reasons),
        event_type_candidates=event_candidates,
    )


def _simhash64(value: str) -> int:
    tokens = _normalize(value).split()
    shingles = [" ".join(tokens[index : index + 5]) for index in range(max(1, len(tokens) - 4))]
    weights = [0] * 64
    for shingle in shingles:
        hashed = int.from_bytes(hashlib.blake2b(shingle.encode(), digest_size=8).digest(), "big")
        for bit in range(64):
            weights[bit] += 1 if hashed & (1 << bit) else -1
    return sum((1 << bit) for bit, weight in enumerate(weights) if weight >= 0)


def _token_similarity(left: str, right: str, *, limit: int | None = None) -> float:
    left_tokens = set(_normalize(left).split()[:limit])
    right_tokens = set(_normalize(right).split()[:limit])
    union = left_tokens | right_tokens
    return len(left_tokens & right_tokens) / len(union) if union else 1.0


def cluster_syndicated_stories(
    articles: Iterable[NewsArticle], *, max_hamming_distance: int = 5
) -> list[ClusteredStory]:
    """Cluster exact and near duplicates with deterministic locality-sensitive bands."""

    ordered = sorted(articles, key=lambda item: (item.provider_timestamp, item.article_id))
    representatives: dict[str, tuple[NewsArticle, int]] = {}
    bands: defaultdict[tuple[int, int], set[str]] = defaultdict(set)
    title_prefixes: defaultdict[tuple[str, ...], set[str]] = defaultdict(set)
    output: list[ClusteredStory] = []
    for article in ordered:
        fingerprint = _simhash64(f"{article.title} {article.content}")
        candidate_clusters: set[str] = set()
        for band in range(4):
            candidate_clusters.update(bands[(band, (fingerprint >> (band * 16)) & 0xFFFF)])
        title_prefix = tuple(_normalize(article.title).split()[:3])
        candidate_clusters.update(title_prefixes[title_prefix])
        cluster_id: str | None = None
        for candidate in sorted(candidate_clusters):
            representative, prior_hash = representatives[candidate]
            within_window = abs(article.provider_timestamp - representative.provider_timestamp) <= timedelta(days=14)
            similar_title_and_body = (
                _token_similarity(article.title, representative.title) >= 0.70
                and _token_similarity(article.content, representative.content, limit=250) >= 0.80
            )
            if within_window and (
                (fingerprint ^ prior_hash).bit_count() <= max_hamming_distance
                or similar_title_and_body
            ):
                cluster_id = candidate
                break
        primary = cluster_id is None
        if cluster_id is None:
            cluster_id = hashlib.sha256(
                f"{article.article_id}:{fingerprint:016x}".encode()
            ).hexdigest()[:20]
            representatives[cluster_id] = (article, fingerprint)
            for band in range(4):
                bands[(band, (fingerprint >> (band * 16)) & 0xFFFF)].add(cluster_id)
            title_prefixes[title_prefix].add(cluster_id)
        output.append(
            ClusteredStory(article_id=article.article_id, cluster_id=cluster_id, primary=primary)
        )
    return output
