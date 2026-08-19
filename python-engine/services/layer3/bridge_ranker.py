import asyncio
import logging
import math
import os
from typing import Any

from schemas.layer3 import (
    IsomorphismMatch,
    MatchType,
    RankedBridge,
    SearchSource,
    Step14Request,
    Step14Response,
    ValidityScores,
)
from services.anthropic_client import evaluate_compatibility_and_transferability

logger = logging.getLogger(__name__)


def _float_env(name: str, default: float) -> float:
    """Safely parse a float from environment variable, falling back to default if invalid."""
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        val = float(raw)
        if not math.isfinite(val):
            logger.warning(
                f"[bridge_ranker] Non-finite float for {name}={raw!r}; using default {default}"
            )
            return default
        return val
    except (ValueError, TypeError):
        logger.warning(
            f"[bridge_ranker] Invalid float for {name}={raw!r}; using default {default}"
        )
        return default


def _sanitize_score(x: Any, default: float = 0.5) -> float:
    """Ensures a score is a finite float strictly bounded between [0.0, 1.0]."""
    try:
        f = float(x)
    except (ValueError, TypeError):
        return default
    if not math.isfinite(f):
        return default
    return min(1.0, max(0.0, f))


def calculate_evidence_strength(match: IsomorphismMatch) -> float:
    """
    Factor 4: Evidence Strength
    Dynamically scores evidence quality using parameterized weights, allowing
    them to be learned or configured without hardcoding magic numbers.
    """
    if (
        not match
        or not match.mechanism
        or not match.mechanism.source_result
        or not match.mechanism.source_result.source_results
    ):
        return 0.05

    source_results = match.mechanism.source_result.source_results

    # Configurable weights (default to calibrated values if not set or invalid)
    weight_patent = _float_env("WEIGHT_PATENT", 0.70)
    weight_paper = _float_env("WEIGHT_PAPER", 0.55)
    weight_wiki = _float_env("WEIGHT_WIKI", 0.45)
    weight_web = _float_env("WEIGHT_WEB", 0.30)

    boost_deployed = _float_env("BOOST_DEPLOYED", 0.25)
    boost_replicated = _float_env("BOOST_REPLICATED", 0.15)
    boost_blog = _float_env("BOOST_BLOG", -0.10)

    cite_500 = _float_env("CITE_500", 0.10)
    cite_100 = _float_env("CITE_100", 0.07)
    cite_10 = _float_env("CITE_10", 0.03)

    rep_5 = _float_env("REP_5", 0.05)
    rep_2 = _float_env("REP_2", 0.02)

    best_score = 0.0
    for result in source_results:
        score = 0.0

        if result.source == SearchSource.PATENT:
            score = weight_patent
        elif result.source == SearchSource.PAPER:
            score = weight_paper
        elif result.source == SearchSource.WIKIPEDIA:
            score = weight_wiki
        elif result.source == SearchSource.WEB:
            score = weight_web

        dep_status = getattr(result, "deployment_status", None)
        if dep_status == "deployed":
            score += boost_deployed
        elif dep_status == "replicated":
            score += boost_replicated
        elif dep_status == "blog":
            score += boost_blog

        cites = getattr(result, "citation_count", 0) or 0
        if cites >= 500:
            score += cite_500
        elif cites >= 100:
            score += cite_100
        elif cites >= 10:
            score += cite_10

        reps = getattr(result, "replication_count", 0) or 0
        if reps >= 5:
            score += rep_5
        elif reps >= 2:
            score += rep_2

        best_score = max(best_score, score)

    return round(min(1.0, max(0.05, best_score)), 3)


async def rank_bridges(request: Step14Request) -> Step14Response:
    """
    Step 14: Bridge Validity Ranker
    Scores surviving matches on 4 factors and returns the Top 3.
    """
    if not request or not request.matches:
        return Step14Response(top_bridges=[])

    # Filter out discarded matches from Step 13
    valid_matches = [m for m in request.matches if m.match_type != MatchType.DISCARDED]

    if not valid_matches:
        return Step14Response(top_bridges=[])

    # Use user domain context if provided, else fall back to top match summary
    domain_context = (request.user_domain_context or "").strip()
    if not domain_context:
        # Sort by isomorphism score to find the best match's summary
        best_match = max(
            valid_matches,
            key=lambda m: (
                m.isomorphism_score
                if m.isomorphism_score is not None
                else 0.0
            ),
        )
        if (
            best_match
            and best_match.mechanism
            and best_match.mechanism.source_result
        ):
            domain_context = (
                best_match.mechanism.source_result.merged_summary or ""
            ).strip()
        logger.warning(
            "[bridge_ranker] WARNING: user_domain_context was empty. "
            "Falling back to top-match summary as LLM context."
        )

    # Truncate domain context to 1500 chars to avoid blowing up context window
    domain_context = domain_context[:1500]

    # Build LLM evaluation tasks using bounded domain context
    tasks = []
    for match in valid_matches:
        candidate_mechanism = ""
        if match.mechanism and match.mechanism.source_result:
            candidate_mechanism = (
                match.mechanism.source_result.merged_summary or ""
            )[:1500]
        tasks.append(
            evaluate_compatibility_and_transferability(
                domain_context=domain_context,
                candidate_mechanism=candidate_mechanism,
            )
        )

    llm_scores = await asyncio.gather(*tasks, return_exceptions=True)

    ranked_bridges = []
    for match, llm_result in zip(valid_matches, llm_scores):
        if isinstance(llm_result, Exception):
            title = (
                match.mechanism.source_result.title
                if match.mechanism and match.mechanism.source_result
                else "unknown"
            )
            logger.warning(
                f"[bridge_ranker] LLM eval failed for bridge '{title}': {llm_result}. "
                "Using neutral 0.5 scores for this match."
            )
            compatibility, transferability = 0.5, 0.5
        elif isinstance(llm_result, (tuple, list)) and len(llm_result) == 2:
            compatibility, transferability = llm_result
        else:
            compatibility, transferability = 0.5, 0.5

        # Sanitize all scores to guarantee finite [0.0, 1.0] range
        raw_iso = (
            match.isomorphism_score
            if match.isomorphism_score is not None
            else 0.0
        )
        structural_match = _sanitize_score(raw_iso / 100.0, default=0.0)
        compatibility = _sanitize_score(compatibility, default=0.5)
        transferability = _sanitize_score(transferability, default=0.5)
        evidence = _sanitize_score(calculate_evidence_strength(match), default=0.05)

        # Final Score = F1 × F2 × F3 × F4 (geometric product)
        final_score = _sanitize_score(
            structural_match * compatibility * transferability * evidence,
            default=0.0,
        )

        scores = ValidityScores(
            structural_match=round(structural_match, 3),
            constraint_compatibility=round(compatibility, 3),
            solution_transferability=round(transferability, 3),
            evidence_strength=round(evidence, 3),
            final_score=round(final_score, 4),
        )

        ranked_bridges.append(RankedBridge(match=match, scores=scores))

    # Sort descending by final score, with tie-breaking on isomorphism_score and evidence_strength
    ranked_bridges.sort(
        key=lambda x: (
            x.scores.final_score,
            x.match.isomorphism_score if x.match.isomorphism_score is not None else 0.0,
            x.scores.evidence_strength,
        ),
        reverse=True,
    )
    return Step14Response(top_bridges=ranked_bridges[:3])
