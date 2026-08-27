"""
services/layer4/context_packager.py — Context Packager for Layer 4

Converts the Top 3 RankedBridge objects into a rich, structured text prompt
that Claude / downstream LLMs can use to generate the FinalReport.

Features:
- Truncation of long summaries and graph descriptions at word boundaries with ellipsis indicators.
- Safe attribute and dictionary access handling missing nodes, edges, or bridge fields without KeyError/IndexError.
- Secret redaction and prompt injection sanitization on all untrusted user and external text.
- Formats multi-layer context into clean, structured prompt context within token limits.
"""
from typing import List, Optional
import re

from schemas.layer3 import RankedBridge

# Secret patterns for redaction
_SECRET_PATTERNS = [
    re.compile(r"sk-ant-[a-zA-Z0-9_\-]{10,}"),
    re.compile(r"sk-[a-zA-Z0-9_\-]{20,}"),
    re.compile(r"(?:ghp|gho|ghu|ghs|ghr)_[a-zA-Z0-9_]{20,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"AIza[0-9A-Za-z\-_]{35}"),
    re.compile(r"(?:bearer\s+)[a-zA-Z0-9_\-\.]{20,}", re.IGNORECASE),
]

_DANGEROUS_TAGS = [
    "<UNTRUSTED_DATA>", "</UNTRUSTED_DATA>",
    "<system>", "</system>",
    "<human>", "</human>",
    "<assistant>", "</assistant>",
    "[INST]", "[/INST]",
    "<<SYS>>", "<</SYS>>",
    "<|im_start|>", "<|im_end|>",
    "<|endoftext|>",
]


def redact_secrets(text: str) -> str:
    """Redacts known API keys and tokens from text to prevent secret leakage."""
    if not text:
        return ""
    redacted = str(text)
    for pattern in _SECRET_PATTERNS:
        redacted = pattern.sub("[REDACTED]", redacted)
    return redacted


def sanitize_prompt_text(text: str) -> str:
    """
    Sanitizes external / untrusted strings to prevent prompt injection and secret leakage.
    Neutralizes delimiter tags and special LLM control tokens.
    """
    if not text:
        return ""
    text = redact_secrets(text)
    for tag in _DANGEROUS_TAGS:
        if tag.lower() in text.lower():
            pattern = re.compile(re.escape(tag), re.IGNORECASE)
            safe_tag = tag.replace("<", "[").replace(">", "]")
            text = pattern.sub(safe_tag, text)
    return text


def _truncate_text(text: str, max_chars: int = 800) -> str:
    """
    Truncates text to `max_chars` at a clean word boundary and appends an ellipsis ('…').
    If the text is shorter than or equal to `max_chars`, returns it unmodified.
    """
    if not text or len(text) <= max_chars:
        return text or ""

    candidate = text[:max_chars]
    last_space = candidate.rfind(" ")
    if last_space > 0:
        candidate = candidate[:last_space]

    return candidate.rstrip() + "…"


def pack_bridges_into_context(bridges: Optional[List[RankedBridge]], user_query: str = "") -> str:
    """
    Converts the Top 3 ranked bridges into a structured, numbered context block
    for the report-generation LLM prompt.

    Each bridge section includes:
    - Title and match classification
    - Quantitative validity scores (all 4 factors)
    - The merged summary (truncated cleanly at word boundary)
    - Causal nodes and edges summary with counts for elided items
    - Any contradiction warnings from the Step 11.5 debate

    Parameters
    ----------
    bridges : Optional[List[RankedBridge]]
        The ranked bridges from Step 14 (at most 3).
    user_query : str
        The original domain-blind structural description for context.

    Returns
    -------
    str
        A human-readable, structured context string ready to be injected
        into the report generation prompt.
    """
    sections = []

    if user_query:
        sanitized_query = sanitize_prompt_text(user_query)
        sanitized_query = _truncate_text(sanitized_query, max_chars=2000)
        sections.append(f"USER PROBLEM (domain-blind structural description):\n{sanitized_query}\n")

    bridge_list = bridges if bridges is not None else []
    sections.append(f"TOP {len(bridge_list)} CROSS-DOMAIN BRIDGES FOUND:\n{'='*60}")

    for rank, bridge in enumerate(bridge_list, 1):
        if bridge is None:
            continue

        match = getattr(bridge, "match", None)
        scores = getattr(bridge, "scores", None)
        mechanism = getattr(match, "mechanism", None) if match else None
        source = getattr(mechanism, "source_result", None) if mechanism else None
        causal_graph = getattr(mechanism, "causal_graph", None) if mechanism else None

        contradiction_block = ""
        contra = getattr(source, "contradiction_analysis", None) if source else None
        if contra and getattr(contra, "conflict_detected", False):
            debate = getattr(contra, "debate_log", None)
            nature = _truncate_text(sanitize_prompt_text(getattr(contra, "nature_of_conflict", "Unknown")), 400)
            proposer = _truncate_text(sanitize_prompt_text(getattr(debate, "proposer_argument", "N/A") if debate else "N/A"), 400)
            skeptic = _truncate_text(sanitize_prompt_text(getattr(debate, "skeptic_rebuttal", "N/A") if debate else "N/A"), 400)
            consensus = _truncate_text(sanitize_prompt_text(getattr(debate, "synthesizer_consensus", "N/A") if debate else "N/A"), 400)
            experiment = _truncate_text(sanitize_prompt_text(getattr(contra, "recommended_experiment", "N/A")), 400)

            contradiction_block = (
                f"\n  ⚠ CONTRADICTION DETECTED:\n"
                f"    Nature: {nature}\n"
                f"    Proposer: {proposer}\n"
                f"    Skeptic: {skeptic}\n"
                f"    Synthesizer consensus: {consensus}\n"
                f"    Recommended experiment: {experiment}"
            )

        if source and getattr(source, "sources", None):
            source_list = ", ".join(getattr(s, "value", str(s)) for s in source.sources)
        else:
            source_list = "N/A"

        all_nodes = getattr(causal_graph, "nodes", None) if causal_graph else None
        if all_nodes:
            nodes_summary = ", ".join(getattr(n, "label", str(n)) for n in all_nodes[:5])
            if len(all_nodes) > 5:
                nodes_summary += f", … (+{len(all_nodes) - 5} more)"
        else:
            nodes_summary = "N/A"

        all_edges = getattr(causal_graph, "edges", None) if causal_graph else None
        if all_edges:
            edges_summary = "; ".join(
                f"{getattr(e, 'source', '')}→{getattr(e, 'target', '')} [{getattr(e, 'relation', '')}]"
                for e in all_edges[:5]
            )
            if len(all_edges) > 5:
                edges_summary += f"; … (+{len(all_edges) - 5} more)"
        else:
            edges_summary = "N/A"

        title = getattr(source, "title", "Untitled Bridge") if source else "Untitled Bridge"
        title = sanitize_prompt_text(title)

        raw_summary = getattr(source, "merged_summary", "") if source else ""
        raw_summary = sanitize_prompt_text(raw_summary)
        summary = _truncate_text(raw_summary, max_chars=800)
        if not summary:
            summary = "N/A"

        match_type_val = "N/A"
        if match and getattr(match, "match_type", None) is not None:
            mt = match.match_type
            match_type_val = getattr(mt, "value", str(mt))

        if scores:
            sm = getattr(scores, "structural_match", 0.0)
            cc = getattr(scores, "constraint_compatibility", 0.0)
            st = getattr(scores, "solution_transferability", 0.0)
            es = getattr(scores, "evidence_strength", 0.0)
            fs = getattr(scores, "final_score", 0.0)
            scores_block = (
                f"  Scores:\n"
                f"    Structural match:         {sm:.2f}\n"
                f"    Constraint compatibility: {cc:.2f}\n"
                f"    Solution transferability: {st:.2f}\n"
                f"    Evidence strength:        {es:.2f}\n"
                f"    FINAL SCORE:              {fs:.4f}"
            )
        else:
            scores_block = (
                f"  Scores:\n"
                f"    Structural match:         0.00\n"
                f"    Constraint compatibility: 0.00\n"
                f"    Solution transferability: 0.00\n"
                f"    Evidence strength:        0.00\n"
                f"    FINAL SCORE:              0.0000"
            )

        section = (
            f"\nBRIDGE #{rank}: {title}\n"
            f"  Match type:     {match_type_val}\n"
            f"  Evidence from:  {source_list}\n"
            f"  Summary:        {summary}\n"
            f"  Causal nodes:   {nodes_summary}\n"
            f"  Causal edges:   {edges_summary}\n"
            f"{scores_block}"
            f"{contradiction_block}"
        )
        sections.append(section)

    return "\n".join(sections)

