import asyncio
import html
import logging
import os
import random
import re
from typing import List, Optional

import httpx

from schemas.layer3 import SearchResult, SearchSource, StructuralQuery
from services.anthropic_client import evaluate_deployment_status

logger = logging.getLogger(__name__)

# Serper.dev API — freemium, as specified in different.md
SERPER_API_URL = "https://google.serper.dev/search"
MAX_RESULTS = 5
DEFAULT_TIMEOUT = 10.0
MAX_RETRIES = 3
MAX_SNIPPET_LENGTH = 10000

# Search query suffix to target engineering whitepapers and case studies globally
SEARCH_SUFFIX = "whitepaper OR case study OR engineering blog"


def _confidence_from_rank(rank: int, total: int) -> float:
    """Top web result = 78%, last = 55%."""
    if total <= 1:
        return 78.0
    return round(78.0 - (rank / max(total - 1, 1)) * 23.0, 1)


def _clean_html_snippet(raw_text: str, max_length: int = MAX_SNIPPET_LENGTH) -> str:
    """
    Unescapes HTML entities, strips script/style/iframe tags and general HTML tags,
    and truncates long text to prevent context overflow or validation errors.
    """
    if not raw_text:
        return ""
    # Strip script, style, and iframe tags along with content
    cleaned = re.sub(
        r"<(script|style|iframe)[^>]*>.*?</\1>",
        "",
        raw_text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    cleaned = re.sub(
        r"<iframe[^>]*>.*?</iframe>|<iframe[^>]*/>|<iframe[^>]*>",
        "",
        cleaned,
        flags=re.DOTALL | re.IGNORECASE,
    )
    # Strip remaining HTML tags
    cleaned = re.sub(r"<[^>]+>", "", cleaned)
    # Decode HTML entities (e.g. &quot;, &amp;, &#39;)
    cleaned = html.unescape(cleaned)
    cleaned = cleaned.strip()
    if len(cleaned) > max_length:
        cleaned = cleaned[:max_length]
    return cleaned


async def _request_with_retry(
    client: httpx.AsyncClient,
    method: str,
    url: str,
    max_retries: int = MAX_RETRIES,
    **kwargs,
) -> httpx.Response:
    """
    Executes an HTTP request with exponential backoff, jitter, and Retry-After header support.
    Retries on 429 (Too Many Requests), 5xx server errors, and network connection/timeout errors.
    """
    last_exc: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            resp = await client.request(method, url, **kwargs)
            if resp.status_code == 429 or resp.status_code >= 500:
                if attempt < max_retries - 1:
                    retry_after = resp.headers.get("Retry-After")
                    if retry_after:
                        try:
                            wait = float(retry_after)
                        except ValueError:
                            wait = 0.5 * (2 ** attempt)
                    else:
                        wait = 0.5 * (2 ** attempt)
                    jitter = random.uniform(0, 0.25)
                    logger.warning(
                        f"[search_web] HTTP {resp.status_code} received on attempt {attempt + 1}. "
                        f"Retrying in {wait + jitter:.2f}s..."
                    )
                    await asyncio.sleep(wait + jitter)
                    continue
            resp.raise_for_status()
            return resp
        except (httpx.RequestError, httpx.HTTPStatusError) as e:
            last_exc = e
            if attempt < max_retries - 1:
                wait = 0.5 * (2 ** attempt) + random.uniform(0, 0.25)
                logger.warning(
                    f"[search_web] Network/HTTP error on attempt {attempt + 1}: {e}. "
                    f"Retrying in {wait:.2f}s..."
                )
                await asyncio.sleep(wait)
            else:
                break
    if last_exc:
        raise last_exc
    raise RuntimeError(f"[search_web] Request failed after {max_retries} retries")


async def search_web(query: StructuralQuery) -> List[SearchResult]:
    """
    Layer 3: Web Search via Serper.dev — Industry whitepapers, Case studies, Tech blogs.

    - Infer deployment_status dynamically using LLM evaluation.
    - Handles rate limits, timeouts, HTML sanitization, entity decoding, and payload truncation.
    - Falls back gracefully to an empty list if SERPER_API_KEY is not configured or on network failures.
    """
    api_key = os.getenv("SERPER_API_KEY", "")
    results: List[SearchResult] = []

    if not api_key:
        logger.warning("[search_web] SERPER_API_KEY not set. Returning empty results.")
        return results

    if not query or not query.structural_description or not query.structural_description.strip():
        return results

    headers = {
        "X-API-KEY": api_key,
        "Content-Type": "application/json",
        "User-Agent": "CausalNexus/1.0 (research bot; contact@causalnexus.ai)",
    }
    payload = {
        "q": f"{query.structural_description.strip()} {SEARCH_SUFFIX}",
        "num": MAX_RESULTS,
    }

    try:
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            response = await _request_with_retry(
                client, "POST", SERPER_API_URL, json=payload, headers=headers
            )
            try:
                data = response.json()
            except Exception as json_err:
                logger.error(f"[search_web] Failed to parse JSON response: {json_err}")
                return results

            if not isinstance(data, dict):
                return results

            organic = data.get("organic", [])
            if not isinstance(organic, list):
                return results

            total = len(organic)

            for rank, item in enumerate(organic):
                if not isinstance(item, dict):
                    continue
                raw_title = item.get("title", "Untitled Article")
                title = _clean_html_snippet(raw_title) or "Untitled Article"

                raw_snippet = item.get("snippet", title)
                snippet = _clean_html_snippet(raw_snippet) or title
                url = item.get("link")

                confidence = _confidence_from_rank(rank, total)

                # ERR-B23 fix: infer status dynamically using an LLM instead of static domains
                deployment_status = "unverified"
                try:
                    deployment_status = await evaluate_deployment_status(url or "", snippet)
                except Exception as llm_err:
                    logger.warning(f"[search_web] evaluate_deployment_status failed: {llm_err}")
                    deployment_status = "unverified"

                results.append(SearchResult(
                    source=SearchSource.WEB,
                    title=title,
                    summary=snippet,
                    url=url,
                    confidence=confidence,
                    original_query=query,
                    citation_count=0,
                    replication_count=0,
                    deployment_status=deployment_status,
                ))

    except (httpx.RequestError, httpx.HTTPStatusError) as exc:
        logger.error(f"[search_web] Serper.dev API error: {exc}. Returning empty results.")
    except Exception as exc:
        logger.error(f"[search_web] Unexpected error in search_web: {exc}. Returning empty results.")

    return results
