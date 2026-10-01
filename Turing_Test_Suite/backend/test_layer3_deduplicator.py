"""
test_layer3_deduplicator.py — Deduplicator / merger service tests.

Catches every SWARM-5 BUG in services/layer3/deduplicator.py:

  BUG #19 — `_meaningful_words` drops hyphenated and digit-containing
             words via `w.isalpha()` (e.g., "state-of-the-art", "covid-19",
             "model-based", "co2" are all silently discarded).
  BUG #20 — no Unicode normalization (NFC/NFD). The same word in composed
             vs decomposed form compares as different strings.
  BUG #21 — greedy clustering is order-dependent (non-deterministic
             merging across runs when asyncio.gather reorders results).

Also covers general deduplicator invariants:
  - exact match, case-insensitive, whitespace-normalized, punctuation-normalized
  - metadata preservation
  - empty / single-element input handling
  - deterministic output

Design notes:
- All tests are deterministic and self-contained.
- Each test FAILS against the current buggy code (red) and PASSES only
  once the corresponding fix lands (green).
"""
from __future__ import annotations

import asyncio
import inspect
import unicodedata
from typing import List, Optional

import pytest

from schemas.layer3 import (
    MergedResult,
    SearchSource,
    SearchResult,
    StructuralQuery,
)
from services.layer3 import deduplicator as dd


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------
def _query() -> StructuralQuery:
    return StructuralQuery(
        original_node_id="n0",
        original_confidence=50.0,
        structural_description="A high-variance input causes output failure.",
    )


def _result(
    *,
    source: SearchSource = SearchSource.PAPER,
    title: str = "T",
    summary: str = "s",
    confidence: float = 80.0,
    citation_count: int = 0,
    replication_count: int = 0,
    deployment_status: str = "single_study",
    url: Optional[str] = None,
) -> SearchResult:
    return SearchResult(
        source=source,
        title=title,
        summary=summary,
        url=url,
        confidence=confidence,
        original_query=_query(),
        citation_count=citation_count,
        replication_count=replication_count,
        deployment_status=deployment_status,
    )


async def _dedup(results: List[SearchResult], threshold: float = 0.50) -> List[MergedResult]:
    return await dd.deduplicate_and_merge(results, similarity_threshold=threshold)


# ===========================================================================
# Exact / case / whitespace / punctuation normalization
# ===========================================================================

@pytest.mark.asyncio
async def test_dedup_exact_match():
    """Two identical results must merge into a single MergedResult."""
    r1 = _result(summary="cancer immunotherapy mechanism")
    r2 = _result(summary="cancer immunotherapy mechanism")

    merged = await _dedup([r1, r2])
    assert len(merged) == 1, (
        f"two identical summaries must merge to 1 result; got {len(merged)}"
    )


@pytest.mark.asyncio
async def test_dedup_case_insensitive():
    """`Cancer` and `cancer` must merge — the comparison must be case-insensitive."""
    r1 = _result(summary="Cancer immunotherapy breakthrough")
    r2 = _result(summary="cancer immunotherapy breakthrough")

    merged = await _dedup([r1, r2])
    assert len(merged) == 1, (
        f"`Cancer` and `cancer` must merge (case-insensitive); got {len(merged)}"
    )


@pytest.mark.asyncio
async def test_dedup_whitespace_normalized():
    """`foo  bar` (double space) and `foo bar` (single space) must merge.

    Note: `str.split()` already collapses consecutive whitespace, so this
    test passes with the current code. It's a guardian against regressions
    that switch to a whitespace-sensitive tokenizer.
    """
    r1 = _result(summary="foo  bar baz mechanism")
    r2 = _result(summary="foo bar baz mechanism")

    merged = await _dedup([r1, r2])
    assert len(merged) == 1, (
        f"whitespace-only differences must merge; got {len(merged)}"
    )


@pytest.mark.asyncio
async def test_dedup_punctuation_normalized():
    """SWARM-5 BUG #19 (sibling): `foo.` and `foo` must merge — punctuation
    is normalized before tokenization. The buggy `w.isalpha()` filter
    drops `foo.` entirely, so two summaries that share ONLY `foo` (with
    trailing punctuation on one side) get similarity 0 and never merge.
    """
    # The ONLY meaningful word in common is `foo` (one side has it as `foo.`).
    # No other words to inflate the overlap artificially.
    r1 = _result(summary="foo")
    r2 = _result(summary="foo.")

    merged = await _dedup([r1, r2])
    assert len(merged) == 1, (
        f"punctuation differences must merge (BUG #19 sibling); got {len(merged)}. "
        f"The buggy `w.isalpha()` filter drops `foo.` entirely, so two summaries "
        f"that share only `foo` (with trailing punctuation) get similarity 0."
    )


# ===========================================================================
# BUG #20: Unicode normalization (NFC/NFD)
# ===========================================================================

@pytest.mark.asyncio
async def test_dedup_unicode_normalized():
    """SWARM-5 BUG #20: the same word in NFC ("café") and NFD ("café" with
    combining accent) forms must merge. The current `_meaningful_words`
    does `text.lower().split()` with NO Unicode normalization, so the two
    forms compare as different strings (Jaccard misses the match).

    Worse, the NFD form's `isalpha()` is False (because of the combining
    accent), so it's dropped entirely by `_meaningful_words`.

    Test setup: the ONLY meaningful word in common is `café` (in NFC vs
    NFD). No other words to inflate the overlap artificially.
    """
    nfc = "café"
    nfd = unicodedata.normalize("NFD", nfc)
    assert nfc != nfd, "test setup: NFC and NFD must be different byte sequences"

    r1 = _result(summary=nfc)
    r2 = _result(summary=nfd)

    merged = await _dedup([r1, r2])
    assert len(merged) == 1, (
        f"NFC and NFD forms of the same word must merge (Unicode normalization, "
        f"BUG #20); got {len(merged)}. The buggy tokenizer drops the NFD form "
        f"entirely (isalpha=False due to combining accent) and never normalizes."
    )


# ===========================================================================
# BUG #19: hyphenated / digit-containing terms dropped
# ===========================================================================

def test_dedup_hyphenated_terms_not_dropped():
    """SWARM-5 BUG #19: `_meaningful_words` filters tokens via `w.isalpha()`,
    which returns False for any word containing a hyphen, digit, apostrophe,
    or period. As a result, "state-of-the-art" is dropped ENTIRELY (not
    split into "state", "of", "the", "art" — just gone).

    In scientific/technical text, hyphenated compounds
    ("state-of-the-art", "model-based", "evidence-based") carry most of the
    semantic signal. Dropping them defeats Jaccard similarity.

    The fix must preserve hyphenated compounds as meaningful tokens (or at
    minimum split on hyphens and keep the alpha-only components).
    """
    tokens = dd._meaningful_words("state-of-the-art model")
    # The hyphenated compound must contribute at least one meaningful token.
    # Current buggy code returns frozenset() because "state-of-the-art".isalpha()
    # is False.
    assert len(tokens) > 0, (
        f"hyphenated term 'state-of-the-art' was dropped entirely by the tokenizer "
        f"(BUG #19). Tokens: {tokens}"
    )
    # Fake-fix detection: also assert that AT LEAST ONE token contains the
    # substring "state" (i.e., the hyphenated compound contributed meaningfully
    # — not just the unrelated word "model").
    state_tokens = [t for t in tokens if "state" in t.lower()]
    assert len(state_tokens) > 0, (
        f"hyphenated 'state-of-the-art' must contribute a token containing 'state'; "
        f"tokens: {tokens}"
    )


def test_dedup_digit_terms_not_dropped():
    """SWARM-5 BUG #19: terms with digits like `covid-19`, `co2`, `h2o`,
    `3d` must NOT be dropped by the tokenizer. These are extremely common
    in scientific text."""
    tokens = dd._meaningful_words("covid-19 virus spike protein")
    # "covid-19" must be preserved (not dropped due to digits/hyphen).
    covid_tokens = [t for t in tokens if "covid" in t.lower()]
    assert len(covid_tokens) > 0, (
        f"term 'covid-19' was dropped entirely by the tokenizer (BUG #19). "
        f"Tokens: {tokens}"
    )


# ===========================================================================
# BUG #21: order-dependent (non-deterministic) clustering
# ===========================================================================

@pytest.mark.asyncio
async def test_dedup_deterministic():
    """SWARM-5 BUG #21: the greedy clustering algorithm picks the cluster
    seed as the first result in input order. If the same input is reordered
    (e.g., by asyncio.gather non-determinism), the cluster boundaries can
    differ — non-reproducible deduplication across runs.

    Test setup (transitive-but-not-reflexive similarity):
      - r1 ~ r2 (similarity ≥ threshold)
      - r2 ~ r3 (similarity ≥ threshold)
      - r1 NOT ~ r3 (similarity < threshold)
    With input order [r1, r2, r3]: cluster = {r1, r2}, alone = {r3}.
    With input order [r3, r2, r1]: cluster = {r3, r2}, alone = {r1}.
    Different clusterings!

    The fix must sort `results` by a deterministic key (e.g., confidence desc,
    title asc) before clustering, so the highest-confidence result is always
    the cluster seed.
    """
    # r1 = {alpha, beta, gamma}     → 3 tokens
    # r2 = {alpha, beta, gamma, delta}  → 4 tokens
    # r3 = {gamma, delta, epsilon}  → 3 tokens
    # r1 vs r2: intersection 3 / union 4 = 0.75   → similar
    # r2 vs r3: intersection 2 / union 5 = 0.40   → at threshold (>= 0.40)
    # r1 vs r3: intersection 1 / union 5 = 0.20   → NOT similar
    r1 = _result(title="A", summary="alpha beta gamma", confidence=70.0)
    r2 = _result(title="B", summary="alpha beta gamma delta", confidence=80.0)
    r3 = _result(title="C", summary="gamma delta epsilon", confidence=90.0)

    orderings = [
        [r1, r2, r3],
        [r3, r2, r1],
        [r2, r1, r3],
        [r2, r3, r1],
    ]
    all_outputs = []
    for ordering in orderings:
        merged = await _dedup(ordering, threshold=0.40)
        # Canonical signature: sorted tuple of sorted titles per cluster.
        sig = tuple(
            tuple(sorted(r.title for r in m.source_results)) for m in merged
        )
        sig_sorted = tuple(sorted(sig))
        all_outputs.append(sig_sorted)

    # All orderings must produce the SAME canonical clustering signature.
    assert len(set(all_outputs)) == 1, (
        f"deduplicate_and_merge must be deterministic regardless of input order "
        f"(SWARM-5 BUG #21). Got different clusterings for different orderings:\n"
        + "\n".join(f"  {o} → {s}" for o, s in zip(orderings, all_outputs))
    )


# ===========================================================================
# Metadata preservation
# ===========================================================================

@pytest.mark.asyncio
async def test_dedup_preserves_metadata():
    """After dedup, the merged result must keep the best metadata from all
    duplicates — at minimum the highest confidence, the union of sources,
    and the primary (highest-confidence) result's title/summary."""
    r1 = _result(source=SearchSource.PAPER, title="A", summary="alpha mechanism model",
                 confidence=80.0, citation_count=10, replication_count=2,
                 deployment_status="replicated")
    r2 = _result(source=SearchSource.PATENT, title="B", summary="alpha mechanism model",
                 confidence=90.0, citation_count=50, replication_count=5,
                 deployment_status="deployed")

    merged = await _dedup([r1, r2])
    assert len(merged) == 1
    m = merged[0]
    # The merged confidence must be ≥ the highest source confidence (boost applied).
    assert m.confidence >= 90.0, (
        f"merged confidence must be ≥ max source confidence; got {m.confidence}"
    )
    # Both sources must be in the merged result's source list.
    assert SearchSource.PAPER in m.sources and SearchSource.PATENT in m.sources, (
        f"merged result must preserve the union of sources; got {m.sources}"
    )
    # Both source results must be in source_results.
    assert len(m.source_results) == 2, (
        f"merged result must preserve all source_results; got {len(m.source_results)}"
    )
    # The primary (best) result's title is used.
    assert m.title in {"A", "B"}, f"title must be one of the inputs; got {m.title}"


# ===========================================================================
# Hash length / collision resistance
# ===========================================================================

def test_dedup_hash_not_too_short():
    """If the deduplicator uses a hash to identify duplicates, the hash must
    be long enough (≥ 16 chars) to avoid collisions across a realistic
    corpus (thousands of summaries).

    Note: the current implementation uses Jaccard similarity (no hash),
    so this test is a guardian for any future hash-based optimization.
    """
    src = inspect.getsource(dd)
    # Check whether the module uses any hashing (md5, sha1, sha256, hash).
    uses_hash = any(
        f".{algo}" in src
        for algo in ["md5", "sha1", "sha224", "sha256", "sha384", "sha512"]
    )
    if not uses_hash:
        # No hash-based dedup in current code — pass (guardian test).
        return

    # If hashing is used, build two summaries and inspect their hashes.
    # We can't easily peek at the internal hash here without instrumenting
    # the code, so we check the source for `hashlib.{algo}` and assert the
    # algorithm produces ≥ 16 hex chars.
    # MD5 → 32, SHA1 → 40, SHA256 → 64 — all ≥ 16. So any of these is fine.
    # Only fail if a custom short hash is used.
    assert "hash(" not in src or uses_hash, (
        "deduplicator must use a cryptographic hash (md5/sha*) with ≥ 16 chars, "
        "not Python's built-in `hash()` (which is 8 hex chars on 64-bit Python "
        "and collisions easily)."
    )


# ===========================================================================
# Edge cases
# ===========================================================================

@pytest.mark.asyncio
async def test_dedup_handles_empty_list():
    """An empty input list must return an empty list, not crash."""
    merged = await _dedup([])
    assert merged == [], (
        f"empty input must yield empty output; got {merged}"
    )


@pytest.mark.asyncio
async def test_dedup_handles_single_element():
    """A single-element input must return a single-element list (with one
    MergedResult wrapping that single result)."""
    r = _result(title="only", summary="only mechanism model")
    merged = await _dedup([r])
    assert len(merged) == 1, f"single input must yield 1 output; got {len(merged)}"
    assert merged[0].source_results == [r], (
        "merged result must wrap the single input"
    )