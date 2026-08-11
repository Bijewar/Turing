import asyncio
import html
import json
import logging
import random
import re
from typing import List, Optional

import httpx

from schemas.layer3 import SearchResult, SearchSource, StructuralQuery

logger = logging.getLogger(__name__)

# Wikipedia Action API — 100% free, no API key required
WIKIPEDIA_SEARCH_API = "https://en.wikipedia.org/w/api.php"
MAX_RESULTS = 3
DEFAULT_TIMEOUT = 10.0
MAX_RETRIES = 3


def _confidence_from_rank(rank: int, total: int) -> float:
    """Top Wikipedia result = 82%, last = 60%."""
    if total <= 1:
        return 82.0
    return round(82.0 - (rank / max(total - 1, 1)) * 22.0, 1)


def _clean_snippet(raw_text: str, max_length: int = 10000) -> str:
    """Unescapes HTML entities, strips script/style/iframe and HTML tags, and truncates length."""
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
                        f"[search_wikipedia] HTTP {resp.status_code} on attempt {attempt + 1}. "
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
                    f"[search_wikipedia] Network/HTTP error on attempt {attempt + 1}: {e}. "
                    f"Retrying in {wait:.2f}s..."
                )
                await asyncio.sleep(wait)
            else:
                break
    if last_exc:
        raise last_exc
    raise RuntimeError(f"[search_wikipedia] Request failed after {max_retries} retries")


async def search_wikipedia(query: StructuralQuery) -> List[SearchResult]:
    """
    Layer 2: Wikipedia REST/Action API
    Fixes Error 1: Makes real HTTP calls to the Wikipedia API with exponential backoff & retry.
    Fixes Error 2: Populates deployment_status='replicated' (Wikipedia articles represent
    established, community-validated knowledge) and citation_count from the article's extlinks.
    Fixes Error 4: Handles redirects, disambiguation pages, missing keys, HTML entities, and pagination.
    """
    results: List[SearchResult] = []

    if not query or not query.structural_description or not query.structural_description.strip():
        return results

    search_params = {
        "action": "query",
        "list": "search",
        "srsearch": query.structural_description.strip(),
        "srlimit": MAX_RESULTS,
        "format": "json",
        "utf8": 1,
    }

    try:
        async with httpx.AsyncClient(
            timeout=DEFAULT_TIMEOUT,
            headers={"User-Agent": "CausalNexus/1.0 (research bot; contact@causalnexus.ai)"},
            follow_redirects=True,
        ) as client:
            articles = []
            curr_params = dict(search_params)

            while len(articles) < MAX_RESULTS:
                search_response = await _request_with_retry(
                    client, "GET", WIKIPEDIA_SEARCH_API, params=curr_params
                )
                try:
                    search_data = search_response.json()
                except Exception as json_err:
                    logger.error(f"[search_wikipedia] Failed to parse JSON response: {json_err}")
                    break

                if not isinstance(search_data, dict):
                    break

                new_articles = search_data.get("query", {}).get("search", [])
                if not new_articles or not isinstance(new_articles, list):
                    break
                articles.extend(new_articles)

                continue_data = search_data.get("continue")
                if not continue_data or not isinstance(continue_data, dict) or len(articles) >= MAX_RESULTS:
                    break
                curr_params = dict(search_params)
                curr_params.update(continue_data)

            articles = articles[:MAX_RESULTS]
            total = len(articles)

            for rank, article in enumerate(articles):
                if not isinstance(article, dict):
                    continue
                title = article.get("title", "")
                raw_snippet = article.get("snippet", title)
                snippet = _clean_snippet(raw_snippet) if raw_snippet else title

                page_id = article.get("pageid")
                url = f"https://en.wikipedia.org/?curid={page_id}" if page_id else None

                # Step 2: Fetch link count and metadata from the article as a proxy for citation_count
                citation_count = 0
                if page_id:
                    info_params = {
                        "action": "query",
                        "prop": "extlinks|pageprops",
                        "pageids": str(page_id),
                        "ellimit": 500,
                        "redirects": 1,
                        "format": "json",
                    }
                    try:
                        info_resp = await _request_with_retry(
                            client, "GET", WIKIPEDIA_SEARCH_API, params=info_params
                        )
                        if info_resp.status_code == 200:
                            info_data = info_resp.json()
                            if isinstance(info_data, dict):
                                pages = info_data.get("query", {}).get("pages", {})
                                page_data = pages.get(str(page_id), {}) if isinstance(pages, dict) else {}
                                
                                # Disambiguation page handling
                                pageprops = page_data.get("pageprops", {}) if isinstance(page_data, dict) else {}
                                if isinstance(pageprops, dict) and "disambiguation" in pageprops:
                                    if not snippet or snippet == title:
                                        snippet = f"{title} (Wikipedia disambiguation page)"

                                extlinks = page_data.get("extlinks", []) if isinstance(page_data, dict) else []
                                if isinstance(extlinks, list):
                                    citation_count = len(extlinks)
                    except Exception as info_err:
                        logger.warning(
                            f"[search_wikipedia] Failed to fetch article metadata for page {page_id}: {info_err}"
                        )

                confidence = _confidence_from_rank(rank, total)

                results.append(SearchResult(
                    source=SearchSource.WIKIPEDIA,
                    title=title or "Untitled Wikipedia Article",
                    summary=snippet or title or "No summary available",
                    url=url,
                    confidence=confidence,
                    original_query=query,
                    citation_count=citation_count,
                    replication_count=0,
                    # Wikipedia articles = community-validated, established knowledge
                    deployment_status="replicated",
                ))

    except (httpx.RequestError, httpx.HTTPStatusError) as exc:
        logger.error(f"[search_wikipedia] Wikipedia API error: {exc}. Returning empty results.")
    except Exception as exc:
        logger.error(f"[search_wikipedia] Unexpected error in search_wikipedia: {exc}. Returning empty results.")

    return results
