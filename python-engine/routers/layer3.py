"""
routers/layer3.py — Layer 3 Cross-Domain Search Endpoints

Exposes the /search, /extract, /match, and /rank endpoints
that wire up the Layer 3 service pipeline with proper rate limiting,
concurrency control, caching, exception isolation, and structured error responses.
"""
import asyncio
import hashlib
import logging
from typing import List, Union

from fastapi import APIRouter, HTTPException, status
import httpx
from cachetools import TTLCache

from schemas.layer3 import (
    Step11Response, Step12Request, Step12Response,
    Step13Request, Step13Response, Step14Request, Step14Response,
    StructuralQuery, MergedResult
)
from services.layer3.search_papers import search_papers
from services.layer3.search_wikipedia import search_wikipedia
from services.layer3.search_web import search_web
from services.layer3.search_patents import search_patents
from services.layer3.deduplicator import deduplicate_and_merge
from services.layer3.contradiction_detector import detect_contradictions
from services.layer3.relation_extractor import run_relation_extraction
from services.layer3.isomorphism import match_graphs
from services.layer3.bridge_ranker import rank_bridges

logger = logging.getLogger(__name__)

router = APIRouter()

# Module-level concurrency semaphores to prevent external API rate-limiting under load
_PAPERS_SEM = asyncio.Semaphore(3)
_WIKI_SEM = asyncio.Semaphore(3)
_WEB_SEM = asyncio.Semaphore(3)
_PATENTS_SEM = asyncio.Semaphore(3)

# Search cache (TTL = 10 minutes, max 512 entries)
_search_cache = TTLCache(maxsize=512, ttl=600)
_cache_lock = asyncio.Lock()

_SOURCE_TIMEOUT_SECONDS = 12.0


def _cache_key(query: StructuralQuery) -> str:
    """Computes a deterministic cache key for a structural search query."""
    key_str = f"{query.original_node_id}:{query.original_confidence}:{query.structural_description.strip().lower()}"
    return hashlib.sha256(key_str.encode("utf-8")).hexdigest()


async def _with_sem(sem: asyncio.Semaphore, coro):
    """Executes a coroutine under the protection of a rate-limiting semaphore."""
    async with sem:
        return await coro


async def _timed_search(coro, source_name: str) -> List:
    """
    Wraps each individual search coroutine in asyncio.wait_for() with a hard deadline.
    Catches network, HTTP, and timeout exceptions gracefully while allowing
    programming errors (NameError, TypeError, AttributeError) to propagate.
    """
    try:
        return await asyncio.wait_for(coro, timeout=_SOURCE_TIMEOUT_SECONDS)
    except asyncio.TimeoutError:
        logger.warning(f"[layer3_search] {source_name} timed out after {_SOURCE_TIMEOUT_SECONDS}s.")
        return []
    except (httpx.RequestError, httpx.HTTPStatusError, httpx.HTTPError) as exc:
        logger.error(f"[layer3_search] {source_name} network/HTTP error: {exc}")
        return []


async def run_step_11_search(query: StructuralQuery) -> Step11Response:
    """
    Step 11 — 4-LAYER PARALLEL SEARCH ENGINE
    Executes concurrent searches across Semantic Scholar (Papers), Wikipedia,
    Serper.dev Web, and Google Patents with per-source semaphores, timeouts,
    in-memory caching, deduplication, and contradiction analysis.
    """
    key = _cache_key(query)
    async with _cache_lock:
        cached = _search_cache.get(key)
    if cached is not None:
        logger.info(f"[layer3_search] Cache hit for query: {query.structural_description[:50]}")
        return cached

    # Run parallel searches across all 4 layers with dedicated concurrency caps
    results_nested = await asyncio.gather(
        _timed_search(_with_sem(_PAPERS_SEM, search_papers(query)), "search_papers"),
        _timed_search(_with_sem(_WIKI_SEM, search_wikipedia(query)), "search_wikipedia"),
        _timed_search(_with_sem(_WEB_SEM, search_web(query)), "search_web"),
        _timed_search(_with_sem(_PATENTS_SEM, search_patents(query)), "search_patents"),
        return_exceptions=True,
    )

    all_results = []
    for res in results_nested:
        if isinstance(res, Exception):
            logger.error(f"[layer3_search] Search source failed with unexpected exception: {res}", exc_info=res)
        elif isinstance(res, list):
            all_results.extend(res)

    merged_results = await deduplicate_and_merge(all_results)
    final_results = await detect_contradictions(merged_results)

    response = Step11Response(
        query=query,
        results=final_results,
    )

    async with _cache_lock:
        _search_cache[key] = response

    return response


@router.post(
    "/search",
    response_model=Step11Response,
    summary="Step 11 — Run 4-layer cross-domain search"
)
async def search(query: StructuralQuery):
    """
    Accepts a domain-blind structural query and runs parallel searches
    across Papers (Semantic Scholar), Wikipedia, Web (Serper), and Patents.
    Returns deduplicated MergedResult objects with optional contradiction analysis.
    """
    try:
        return await run_step_11_search(query)
    except HTTPException:
        raise
    except ValueError as exc:
        logger.warning(f"[layer3 /search] Validation error: {exc}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    except Exception as exc:
        logger.exception(f"[layer3 /search] Unhandled error: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during search execution. See server logs for details."
        )


@router.post(
    "/extract",
    response_model=Step12Response,
    summary="Step 12 — Extract causal graphs from search results"
)
async def extract(request: Union[Step11Response, Step12Request]):
    """
    Accepts the Step 11 response or Step 12 request and runs LLM-based causal graph extraction
    on each MergedResult. Returns a list of ExtractedMechanism objects.
    """
    try:
        if isinstance(request, Step11Response):
            results = request.results
        elif isinstance(request, Step12Request):
            results = request.merged_results
        else:
            results = getattr(request, "results", getattr(request, "merged_results", []))

        mechanisms = await run_relation_extraction(results)
        return Step12Response(extracted_mechanisms=mechanisms)
    except HTTPException:
        raise
    except ValueError as exc:
        logger.warning(f"[layer3 /extract] Validation error: {exc}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    except Exception as exc:
        logger.exception(f"[layer3 /extract] Unhandled error: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during causal graph extraction. See server logs for details."
        )


@router.post(
    "/match",
    response_model=Step13Response,
    summary="Step 13 — Graph isomorphism matching"
)
async def match(request: Step13Request):
    """
    Compares each candidate ExtractedMechanism against the user's target graph
    using structural isomorphism scoring. Returns IsomorphismMatch objects
    classified as PERFECT / STRONG_PARTIAL / WEAK_PARTIAL / DISCARDED.
    """
    try:
        return await match_graphs(request)
    except HTTPException:
        raise
    except ValueError as exc:
        logger.warning(f"[layer3 /match] Validation error: {exc}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    except Exception as exc:
        logger.exception(f"[layer3 /match] Unhandled error: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during isomorphism matching. See server logs for details."
        )


@router.post(
    "/rank",
    response_model=Step14Response,
    summary="Step 14 — Bridge validity ranking"
)
async def rank(request: Step14Request):
    """
    Scores surviving matches on 4 validity factors (structural match,
    constraint compatibility, solution transferability, evidence strength)
    and returns the Top 3 cross-domain bridges.
    """
    try:
        return await rank_bridges(request)
    except HTTPException:
        raise
    except ValueError as exc:
        logger.warning(f"[layer3 /rank] Validation error: {exc}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    except Exception as exc:
        logger.exception(f"[layer3 /rank] Unhandled error: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during bridge ranking. See server logs for details."
        )
