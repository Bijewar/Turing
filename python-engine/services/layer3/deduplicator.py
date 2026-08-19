from __future__ import annotations

import hashlib
import re
from typing import List, Optional
import unicodedata

from schemas.graph import CausalGraph
from schemas.layer3 import SearchResult, MergedResult, SearchSource

# ---------------------------------------------------------------------------
# Configurable confidence boost per corroborating source type.
# Reflects the epistemic value of evidence from each domain — a patent +
# paper is worth more than 2 web results.
# ---------------------------------------------------------------------------
EVIDENCE_BOOST_MAP = {
    SearchSource.PATENT:    5.0,   # Deployed engineering solution → highest boost
    SearchSource.PAPER:     4.0,   # Peer-reviewed proof → very high boost
    SearchSource.WIKIPEDIA: 2.5,   # Community-validated knowledge → medium boost
    SearchSource.WEB:       1.0,   # Whitepapers / blogs → low boost
}

# ---------------------------------------------------------------------------
# Common English stopwords stripped before Jaccard similarity.
# Without this, words like "the", "of", "is" inflate similarity between
# completely unrelated summaries and cause over-merging (L3-4 fix).
# ---------------------------------------------------------------------------
_STOPWORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "dare", "used",
    "of", "in", "to", "for", "on", "at", "by", "with", "from", "as",
    "and", "or", "but", "not", "this", "that", "it", "its", "they",
    "their", "which", "who", "when", "where", "how", "what", "all",
    "also", "so", "into", "than", "then", "there", "these", "those",
    "such", "both", "each", "more", "other", "no", "nor", "up", "out",
})

# Tokenizer preserving unicode characters, digits, internal hyphens and apostrophes
_TOKEN_RE = re.compile(r"[\w][\w'\-]*[\w]|[\w]", re.UNICODE)
MAX_MERGED_SUMMARY_LEN = 2000


def normalize_node_label(label: str) -> str:
    """Normalizes a node label with NFC Unicode normalization and whitespace trimming."""
    if not label:
        return ""
    normalized = unicodedata.normalize("NFC", label).strip().lower()
    return " ".join(normalized.split())


def canonical_graph_representation(graph: CausalGraph) -> str:
    """Generates a deterministic canonical string representation of a CausalGraph."""
    sorted_nodes = sorted(
        (normalize_node_label(n.label), n.domain or "", round(n.confidence, 2))
        for n in graph.nodes
    )
    sorted_edges = sorted(
        (
            normalize_node_label(e.source),
            normalize_node_label(e.target),
            e.relation.strip().upper(),
            round(e.confidence, 2),
            round(e.weight, 4),
        )
        for e in graph.edges
    )
    return f"NODES:{sorted_nodes}|EDGES:{sorted_edges}"


def compute_graph_hash(graph: CausalGraph) -> str:
    """Computes a collision-resistant SHA-256 hash (≥16 hex characters) for a graph."""
    canonical_repr = canonical_graph_representation(graph)
    return hashlib.sha256(canonical_repr.encode("utf-8")).hexdigest()[:32]


def _meaningful_words(text: str) -> frozenset:
    """Tokenises text and strips stopwords so only domain-meaningful words
    contribute to the Jaccard similarity score.
    Preserves hyphenated words, digits, and Unicode characters."""
    if not text:
        return frozenset()
    text = unicodedata.normalize("NFC", text).lower()
    return frozenset(
        w for w in (m.group(0) for m in _TOKEN_RE.finditer(text))
        if w not in _STOPWORDS
    )


def _compute_similarity(a: SearchResult, b: SearchResult) -> float:
    """Calculates stopword-filtered Jaccard similarity between two result summaries.
    Symmetric and strictly bounded in [0.0, 1.0]."""
    words_a = _meaningful_words(a.summary)
    words_b = _meaningful_words(b.summary)

    if not words_a or not words_b:
        return 0.0

    intersection = words_a & words_b
    union = words_a | words_b
    return len(intersection) / len(union)


async def deduplicate_and_merge(
    results: List[SearchResult],
    similarity_threshold: float = 0.50,
) -> List[MergedResult]:
    """Real semantic deduplication using stopword-filtered Jaccard similarity.
    Deterministic clustering seeded by confidence descending, title ascending."""
    if not results:
        return []

    # Deterministically sort results before clustering to prevent non-deterministic
    # cluster formation across runs
    sorted_results = sorted(
        results,
        key=lambda r: (-r.confidence, r.title, r.summary, r.url or ""),
    )

    visited = [False] * len(sorted_results)
    groups: List[List[SearchResult]] = []

    for i, result_i in enumerate(sorted_results):
        if visited[i]:
            continue
        group = [result_i]
        visited[i] = True
        for j, result_j in enumerate(sorted_results):
            if visited[j]:
                continue
            similarity = _compute_similarity(result_i, result_j)
            if similarity >= similarity_threshold:
                group.append(result_j)
                visited[j] = True
        groups.append(group)

    # Build MergedResult for each cluster
    merged_results: List[MergedResult] = []
    for group in groups:
        sources = list({r.source for r in group})
        base_confidence = max(r.confidence for r in group)

        # Configurable evidence boost based on source type epistemic value
        boost = sum(EVIDENCE_BOOST_MAP.get(s, 0.5) for s in sources)
        final_confidence = min(100.0, max(0.0, base_confidence + boost))

        # Use the highest-confidence result as primary
        primary = max(group, key=lambda r: (r.confidence, len(r.summary)))

        # Bounded merged summary concatenation to avoid blowing LLM context
        merged_summary = primary.summary
        if len(group) > 1:
            other_summaries = [r.summary for r in group if r is not primary]
            kept: List[str] = []
            prefix = " [Corroborated by: "
            suffix = "]"
            running_len = len(merged_summary) + len(prefix) + len(suffix)

            for s in other_summaries:
                addition_len = len(s) + (3 if kept else 0)  # " | " separator
                if running_len + addition_len > MAX_MERGED_SUMMARY_LEN:
                    break
                kept.append(s)
                running_len += addition_len

            if kept:
                if len(kept) < len(other_summaries):
                    merged_summary += f"{prefix}{' | '.join(kept)} | ...{suffix}"
                else:
                    merged_summary += f"{prefix}{' | '.join(kept)}{suffix}"

        merged = MergedResult(
            title=primary.title,
            merged_summary=merged_summary,
            underlying_mechanism=primary.original_query.structural_description,
            sources=sources,
            source_results=group,
            confidence=final_confidence,
        )
        merged_results.append(merged)

    return merged_results
