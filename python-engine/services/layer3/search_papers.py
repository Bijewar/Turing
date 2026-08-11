import asyncio
import logging
import math
import os
import random
from typing import List, Optional

import httpx
from schemas.layer3 import SearchResult, SearchSource, StructuralQuery

logger = logging.getLogger(__name__)

# Semantic Scholar Graph API — 100% free for academic use, no key required
SEMANTIC_SCHOLAR_API = "https://api.semanticscholar.org/graph/v1/paper/search"
FIELDS = "title,abstract,citationCount,year,url,externalIds,authors"
MAX_RESULTS = 5


def _deployment_status_from_citations(citation_count: int) -> str:
    """Derives deployment status from citation count as a replication proxy."""
    try:
        c_count = int(citation_count)
    except (ValueError, TypeError):
        c_count = 0
    if c_count >= 500:
        return "deployed"
    elif c_count >= 50:
        return "replicated"
    elif c_count >= 5:
        return "single_study"
    else:
        return "unverified"


def _confidence_from_rank(rank: int, citation_count: int) -> float:
    """
    Calculates confidence using a grounded metric combining rank decay 
    and logarithmic citation validation.
    """
    try:
        r = max(int(rank), 0)
        c_count = max(int(citation_count), 0)
    except (ValueError, TypeError):
        r = 0
        c_count = 0
    base_conf = 100.0 / (1.0 + 0.5 * r)
    cite_boost = 10.0 * math.log10(max(c_count, 1))
    return round(min(99.9, max(0.0, base_conf + cite_boost)), 1)


async def _request_with_retry(
    client: httpx.AsyncClient,
    method: str,
    url: str,
    max_retries: int = 3,
    **kwargs,
) -> httpx.Response:
    """
    Executes an HTTP request with exponential backoff on 429 (rate-limit) and 5xx (server errors),
    honoring the Retry-After header when present.
    """
    last_exc: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            resp = await client.request(method, url, **kwargs)
            if resp.status_code == 429:
                if attempt < max_retries - 1:
                    retry_after = resp.headers.get("Retry-After")
                    try:
                        wait = float(retry_after) if retry_after else (0.5 * (2 ** attempt))
                    except (ValueError, TypeError):
                        wait = 0.5 * (2 ** attempt)
                    await asyncio.sleep(wait + random.uniform(0, 0.25))
                    continue
                else:
                    resp.raise_for_status()
            elif resp.status_code >= 500:
                if attempt < max_retries - 1:
                    await asyncio.sleep(0.5 * (2 ** attempt) + random.uniform(0, 0.25))
                    continue
                else:
                    resp.raise_for_status()
            resp.raise_for_status()
            return resp
        except (httpx.RequestError, httpx.HTTPStatusError) as exc:
            last_exc = exc
            if attempt < max_retries - 1:
                await asyncio.sleep(0.5 * (2 ** attempt) + random.uniform(0, 0.25))
            else:
                raise
    if last_exc:
        raise last_exc
    raise RuntimeError("Request failed after retries")


async def search_papers(query: StructuralQuery) -> List[SearchResult]:
    """
    Layer 1: SPECTER2 / Semantic Scholar Graph API
    Fixes Error 3: Makes a real HTTP call to the Semantic Scholar API.
    Fixes Error 4: Populates citation_count and deployment_status from actual API response data.
    Fixes BUG #11: Implements exponential backoff and retry on HTTP 429/5xx.
    Fixes BUG #12: Configures User-Agent and optional S2_API_KEY.
    Fixes BUG #13: Safely casts citation_count to int to avoid TypeErrors.
    Fixes BUG #14: Sets replication_count to 0 (never fabricates from citation count).

    Uses the domain-blind structural_description as the search query
    so results are cross-domain by design.
    """
    if not query or not getattr(query, "structural_description", None):
        return []

    base_params = {
        "query": query.structural_description,
        "fields": FIELDS,
        "limit": MAX_RESULTS,
    }

    headers = {
        "User-Agent": "CausalNexus/1.0 (research bot; contact@causalnexus.ai)",
        "Accept": "application/json",
    }
    s2_key = os.getenv("S2_API_KEY", "").strip()
    if s2_key:
        headers["x-api-key"] = s2_key

    limits = httpx.Limits(max_keepalive_connections=5, max_connections=10)
    timeout = httpx.Timeout(15.0, connect=5.0)

    results: List[SearchResult] = []

    try:
        async with httpx.AsyncClient(headers=headers, timeout=timeout, limits=limits, follow_redirects=True) as client:
            papers_list: List[dict] = []
            next_token: Optional[str] = None
            max_pages = 3

            while len(papers_list) < MAX_RESULTS and max_pages > 0:
                max_pages -= 1
                params = dict(base_params)
                if next_token:
                    params["token"] = next_token

                response = await _request_with_retry(client, "GET", SEMANTIC_SCHOLAR_API, params=params)
                try:
                    data = response.json()
                except Exception as json_err:
                    logger.error(f"[search_papers] JSON decode error from Semantic Scholar: {json_err}")
                    break

                if not isinstance(data, dict):
                    logger.warning(f"[search_papers] Unexpected response structure from Semantic Scholar: {type(data)}")
                    break

                page_papers = data.get("data", [])
                if isinstance(page_papers, list):
                    for p in page_papers:
                        if isinstance(p, dict):
                            papers_list.append(p)

                next_token = data.get("next") or data.get("token")
                if not next_token or len(papers_list) >= MAX_RESULTS:
                    break

            for rank, paper in enumerate(papers_list[:MAX_RESULTS]):
                title_raw = paper.get("title")
                title = str(title_raw).strip() if title_raw and isinstance(title_raw, str) else "Untitled Paper"

                abstract_raw = paper.get("abstract")
                abstract = str(abstract_raw).strip() if abstract_raw and isinstance(abstract_raw, str) else title

                try:
                    citation_count = int(paper.get("citationCount") or 0)
                    if citation_count < 0:
                        citation_count = 0
                except (ValueError, TypeError):
                    citation_count = 0

                external_ids = paper.get("externalIds")
                if not isinstance(external_ids, dict):
                    external_ids = {}

                # Build URL from DOI, arXiv ID, or CorpusId if available
                url = paper.get("url")
                if not url or not isinstance(url, str):
                    doi = external_ids.get("DOI") if isinstance(external_ids, dict) else None
                    arxiv_id = external_ids.get("ArXiv") if isinstance(external_ids, dict) else None
                    corpus_id = external_ids.get("CorpusId") if isinstance(external_ids, dict) else None
                    if doi:
                        url = f"https://doi.org/{doi}"
                    elif arxiv_id:
                        url = f"https://arxiv.org/abs/{arxiv_id}"
                    elif corpus_id:
                        url = f"https://www.semanticscholar.org/paper/{corpus_id}"
                    else:
                        url = None

                confidence = _confidence_from_rank(rank, citation_count)
                deployment_status = _deployment_status_from_citations(citation_count)
                # BUG #14 fix: Do NOT fabricate replication counts from raw citation counts.
                replication_count = 0

                results.append(SearchResult(
                    source=SearchSource.PAPER,
                    title=title,
                    summary=abstract,
                    url=url,
                    confidence=confidence,
                    original_query=query,
                    citation_count=citation_count,
                    replication_count=replication_count,
                    deployment_status=deployment_status,
                ))

    except (httpx.RequestError, httpx.HTTPStatusError) as exc:
        # Graceful fallback — log the failure and return empty list.
        # The pipeline uses return_exceptions=True so this won't crash the other 3 layers.
        logger.error(f"[search_papers] Semantic Scholar API error: {exc}. Returning empty results.")
    except Exception as exc:
        logger.error(f"[search_papers] Unexpected error during paper search: {exc}", exc_info=True)

    return results
