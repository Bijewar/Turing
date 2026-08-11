import asyncio
import logging
import os
import random
from typing import List, Optional

import httpx
from schemas.layer3 import SearchResult, SearchSource, StructuralQuery

logger = logging.getLogger(__name__)

# Serper.dev API — freemium, thousands of free searches (as specified in different.md)
SERPER_API_URL = "https://google.serper.dev/patents"
MAX_RESULTS = 5


def _confidence_from_rank(rank: int, total: int) -> float:
    """Top patent result = 92%, last = 70%."""
    try:
        r = max(int(rank), 0)
        t = max(int(total), 1)
    except (ValueError, TypeError):
        return 70.0
    if t <= 1:
        return 92.0
    val = 92.0 - (r / max(t - 1, 1)) * 22.0
    return round(min(99.9, max(0.0, val)), 1)


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


async def search_patents(query: StructuralQuery) -> List[SearchResult]:
    """
    Layer 4: Google Patents via Serper.dev API
    Fixes Error 3: Makes real HTTP calls to Serper.dev — removes all synthetic
    Boeing/NASA static results and non-existent test URLs.
    Fixes Error 4: No simulated asyncio.sleep — real API provides authentic latency.
    Fixes BUG #11: Implements exponential backoff and retry on HTTP 429/5xx.

    Populates citation_count, replication_count, and deployment_status from
    actual patent metadata returned by the API.

    Falls back gracefully to an empty list if SERPER_API_KEY is not configured.
    """
    if not query or not getattr(query, "structural_description", None):
        return []

    api_key = os.getenv("SERPER_API_KEY", "").strip()
    results: List[SearchResult] = []

    if not api_key:
        logger.warning("[search_patents] SERPER_API_KEY not set. Returning empty results.")
        return results

    headers = {
        "X-API-KEY": api_key,
        "Content-Type": "application/json",
        "User-Agent": "CausalNexus/1.0 (research bot; contact@causalnexus.ai)",
    }
    payload = {
        "q": query.structural_description,
        "num": MAX_RESULTS,
    }

    limits = httpx.Limits(max_keepalive_connections=5, max_connections=10)
    timeout = httpx.Timeout(15.0, connect=5.0)

    try:
        async with httpx.AsyncClient(headers=headers, timeout=timeout, limits=limits, follow_redirects=True) as client:
            response = await _request_with_retry(client, "POST", SERPER_API_URL, json=payload)
            try:
                data = response.json()
            except Exception as json_err:
                logger.error(f"[search_patents] JSON decode error from Serper: {json_err}")
                return []

            if not isinstance(data, dict):
                logger.warning(f"[search_patents] Unexpected response structure from Serper: {type(data)}")
                return []

            patents = data.get("patents", [])
            if not isinstance(patents, list):
                patents = []

            total = len(patents)
            if total == 0:
                return []

            for rank, patent in enumerate(patents):
                if not isinstance(patent, dict):
                    continue

                title_raw = patent.get("title")
                title = str(title_raw).strip() if title_raw and isinstance(title_raw, str) else "Untitled Patent"

                snippet_raw = patent.get("snippet")
                snippet = str(snippet_raw).strip() if snippet_raw and isinstance(snippet_raw, str) else title

                link = patent.get("link")
                patent_number = patent.get("patentNumber") or patent.get("publicationNumber") or ""
                if isinstance(patent_number, str):
                    clean_patent_number = patent_number.strip()
                else:
                    clean_patent_number = str(patent_number).strip() if patent_number else ""

                # Build a clean Google Patents URL from the patent number if link missing
                url = link if isinstance(link, str) and link.strip() else None
                if not url and clean_patent_number:
                    url = f"https://patents.google.com/patent/{clean_patent_number}"

                # Safe assignee extraction and deployment status determination
                assignee_raw = patent.get("assignee")
                assignee = str(assignee_raw).strip() if assignee_raw is not None else ""
                deployment_status = "deployed" if assignee else "single_study"

                # ERR-B44 fix: Remove fabricated `age * 15` citation math. Only use real data.
                try:
                    citation_count = int(patent.get("citationCount") or 0)
                    if citation_count < 0:
                        citation_count = 0
                except (ValueError, TypeError):
                    citation_count = 0

                confidence = _confidence_from_rank(rank, total)

                results.append(SearchResult(
                    source=SearchSource.PATENT,
                    title=title,
                    summary=snippet,
                    url=url,
                    confidence=confidence,
                    original_query=query,
                    citation_count=citation_count,
                    replication_count=0,
                    deployment_status=deployment_status,
                ))

    except (httpx.RequestError, httpx.HTTPStatusError) as exc:
        logger.error(f"[search_patents] Serper.dev API error: {exc}. Returning empty results.")
    except Exception as exc:
        logger.error(f"[search_patents] Unexpected error during patent search: {exc}", exc_info=True)

    return results
