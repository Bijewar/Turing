
import io
import json
import logging
import os
import statistics
import re
from typing import Dict, Any, Tuple, List

import pandas as pd
import pdfplumber

logger = logging.getLogger(__name__)

# ==============================================================================
# 1. SIMPLE MATRIX EXTRACTION (Mayank)
# ==============================================================================

def extract_numeric_features(content: dict) -> Tuple[List[str], List[List[float]]]:
    """
    Extracts only the numeric columns from a parsed tabular content dict
    and returns them as a list of column names and a 2D list of float values.

    Non-numeric columns (strings, dates) are silently dropped.

    Parameters
    ----------
    content : dict
        Output of universal_parser._parse_tabular(), with keys:
        "columns", "rows", "row_count".

    Returns
    -------
    Tuple[List[str], List[List[float]]]
        (numeric_columns, data_matrix)
        where data_matrix is rows × columns, indexed as data_matrix[row][col].
    """
    columns: List[str] = content.get("columns", [])
    rows: List[list] = content.get("rows", [])

    if not rows:
        return [], []

    # Identify numeric column indices
    numeric_col_indices = []
    for col_idx, col_name in enumerate(columns):
        # Test the first non-None value in this column
        for row in rows:
            val = row[col_idx] if col_idx < len(row) else None
            if val is not None:
                try:
                    float(val)
                    numeric_col_indices.append(col_idx)
                except (TypeError, ValueError):
                    pass  # Skip non-numeric columns
                break

    if not numeric_col_indices:
        return [], []

    numeric_columns = [columns[i] for i in numeric_col_indices]

    # Build data matrix — replace None/non-numeric with column mean
    raw_columns: List[List[float]] = []
    for col_idx in numeric_col_indices:
        vals = []
        for row in rows:
            val = row[col_idx] if col_idx < len(row) else None
            try:
                vals.append(float(val))
            except (TypeError, ValueError):
                vals.append(None)
        # Fill missing values with column mean
        valid = [v for v in vals if v is not None]
        col_mean = statistics.mean(valid) if valid else 0.0
        raw_columns.append([v if v is not None else col_mean for v in vals])

    # Transpose to row-major order: data_matrix[row][col]
    num_rows = len(rows)
    data_matrix = [
        [raw_columns[col_i][row_i] for col_i in range(len(numeric_col_indices))]
        for row_i in range(num_rows)
    ]

    return numeric_columns, data_matrix


def extract_summary_statistics(content: dict) -> Dict[str, dict]:
    """
    Computes basic summary statistics for each numeric column.

    Returns
    -------
    Dict[str, dict]
        Maps column name → {"mean": float, "std": float, "min": float, "max": float}
    """
    columns, matrix = extract_numeric_features(content)

    if not columns:
        return {}

    stats: Dict[str, dict] = {}
    for col_idx, col_name in enumerate(columns):
        col_vals = [matrix[row_i][col_idx] for row_i in range(len(matrix))]
        if not col_vals:
            continue
        stats[col_name] = {
            "mean": round(statistics.mean(col_vals), 4),
            "std": round(statistics.stdev(col_vals) if len(col_vals) > 1 else 0.0, 4),
            "min": round(min(col_vals), 4),
            "max": round(max(col_vals), 4),
        }

    return stats


# ==============================================================================
# 2. PANDAS EXTRACTION & AMBIGUITY DETECTOR (Harsh)
# ==============================================================================

class UniversalExtractor:
    """
    Step 2: Universal Data Extractor.
    Takes a file payload and standardizes it into either a clean DataFrame
    (DATA path) or a normalized string (TEXT path).
    """

    @staticmethod
    def extract_data(file_bytes: bytes, filename: str) -> Tuple[pd.DataFrame, list]:
        """
        Rips numbers out of files (CSV, Excel, JSON, PDF tables) and forces them
        into a clean pandas DataFrame. Applies strict mathematical validation.
        """
        ext = filename.split('.')[-1].lower() if '.' in filename else ''
        df = pd.DataFrame()
        
        try:
            if ext in ['csv']:
                # low_memory=False forces pandas to read the ENTIRE file before
                # deciding column dtypes. Without this, large CSVs are read in
                # chunks and mixed-type columns (floats + some strings) end up as
                # 'object' dtype, causing the "No numeric columns found" error.
                df = pd.read_csv(
                    io.BytesIO(file_bytes),
                    low_memory=False,
                    encoding_errors='replace',  # handle non-UTF-8 characters gracefully
                )
            elif ext in ['xlsx', 'xls']:
                df = pd.read_excel(io.BytesIO(file_bytes))
            elif ext in ['json']:
                df = pd.read_json(io.BytesIO(file_bytes))
            elif ext in ['pdf']:
                df = UniversalExtractor._extract_tables_from_pdf(file_bytes)
            else:
                raise ValueError(
                    f"Unsupported file extension '.{ext}'. "
                    "Supported data formats: csv, xlsx, xls, json, pdf."
                )
        except Exception as e:
            logger.error(f"Failed to parse {filename} as data: {e}")
            raise ValueError(f"Could not parse file as tabular data. Error: {e}")

        # Standardize and Validate
        return UniversalExtractor.validate_data(df)

    @staticmethod
    def extract_text(file_bytes: bytes, filename: str) -> str:
        """
        Extracts unstructured text from files (PDF, TXT, MD) into a single clean string.
        Applies basic cleaning to normalize whitespace. (Note: chunking is handled in Step 3).
        """
        ext = filename.split('.')[-1].lower() if '.' in filename else ''
        
        def _clean(raw_text: str) -> str:
            # Replace multiple newlines with a single newline
            cleaned = re.sub(r'\n{2,}', '\n', raw_text)
            # Replace multiple spaces with a single space
            cleaned = re.sub(r'[ \t]+', ' ', cleaned)
            return cleaned.strip()

        if ext in ['pdf']:
            text = ""
            try:
                with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
                return _clean(text)
            except Exception as e:
                logger.error(f"Failed to extract text from PDF: {e}")
                raise ValueError("Could not extract text from PDF.")
        else:
            try:
                raw_text = file_bytes.decode('utf-8')
                return _clean(raw_text)
            except UnicodeDecodeError:
                raise ValueError("File is not a valid UTF-8 text file.")

    @staticmethod
    def _extract_tables_from_pdf(file_bytes: bytes) -> pd.DataFrame:
        """
        Combines all tables from a PDF into a single DataFrame.
        """
        combined_data = []
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for page in pdf.pages:
                tables = page.extract_tables()
                for table in tables:
                    combined_data.extend(table)
                    
        if not combined_data:
            raise ValueError("No tables found in PDF.")
            
        # Treat first row as header
        df = pd.DataFrame(combined_data[1:], columns=combined_data[0])
        
        # Coerce columns to numeric where possible.
        # errors='coerce' converts invalid values to NaN rather than ignoring the
        # whole column — this correctly handles mixed columns (numbers + some strings).
        for col in df.columns:
            converted = pd.to_numeric(df[col], errors='coerce')
            # Only apply conversion if at least 50% of values are numeric
            # (avoids turning a true text column into a NaN-filled numeric column)
            if converted.notna().mean() >= 0.5:
                df[col] = converted
            
        return df

    @staticmethod
    def validate_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, list]:
        """
        Applies mathematical constraints to the tabular data.
        Returns the cleaned DataFrame and a list of warnings.
        """
        warnings = []
        
        if df.empty:
            raise ValueError("The extracted dataset is completely empty.")

        # 1. Drop columns that are completely all-NaN first
        df = df.dropna(axis=1, how='all')

        # 2. For any object-dtype columns, try to coerce to numeric.
        # This handles files where numbers were read as strings due to mixed data.
        for col in df.select_dtypes(include=['object']).columns:
            converted = pd.to_numeric(df[col], errors='coerce')
            if converted.notna().mean() >= 0.5:  # >50% numeric values → treat as numeric
                df[col] = converted

        # 3. Drop rows with any missing values in remaining columns to ensure pristine math for PC Algorithm
        initial_len = len(df)
        df = df.dropna()
        dropped = initial_len - len(df)
        if dropped > 0:
            warnings.append(f"Dropped {dropped} rows containing missing values (NaN) to maintain mathematical integrity.")

        # 4. Keep only numeric columns
        numeric_df = df.select_dtypes(include=['number'])
        if numeric_df.empty:
            raise ValueError(
                "No numeric columns found after processing. "
                "The file appears to contain only text data. "
                "Please upload a CSV where at least some columns contain numbers."
            )

        # 4.1 Drop zero-variance (constant) columns
        variances = numeric_df.var()
        constant_cols = variances[variances == 0].index
        if len(constant_cols) > 0:
            numeric_df = numeric_df.drop(columns=constant_cols)
            warnings.append(f"Dropped {len(constant_cols)} constant columns.")

        # 4.2 Add tiny jitter to prevent singular matrix errors in causal discovery
        import numpy as np
        seed = int(os.getenv("CAUSAL_RANDOM_STATE", os.getenv("TURING_RANDOM_SEED", "42")))
        rng = np.random.default_rng(seed)
        numeric_df = numeric_df + rng.normal(0, 1e-6, numeric_df.shape)

        # 5. Check for low data warning
        min_rows = int(os.getenv("MIN_DATA_ROWS", "30"))
        final_len = len(numeric_df)
        if final_len < min_rows:
            warnings.append(
                f"LOW_DATA_WARNING: Dataset has only {final_len} rows. "
                f"Causal discovery confidence may be low. "
                f"Minimum {min_rows} rows recommended."
            )

        return numeric_df, warnings


# Penalty weights (all additive, capped at 100)
_PENALTY_LOW_EDGE_CONF = 20       # any adjacent edge has confidence < 0.60
_PENALTY_HIDDEN_NODE = 30         # node is inferred / hidden
_PENALTY_CONTRADICTED_EDGE = 25   # node has at least one contradicted edge
_PENALTY_BAD_FIT = 15             # GP fit failed or is missing for DATA path
_PENALTY_LOW_DEGREE = 10          # node has only 1 connection total


class AmbiguityDetector:
    """
    Step 7 of the Layer 1 pipeline.
    Adds ambiguity/urgency scores to each node and an overall confidence score
    to the graph dict so that ConfidenceChecker (Step 8) can make routing
    decisions without re-traversing the topology.
    """

    @staticmethod
    def analyze_graph(graph_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Annotates every node in ``graph_data`` with an ``ambiguity_score``
        and ``urgency_score`` (0–100, higher = more uncertain), then writes:

        - graph_data["overall_graph_confidence"] — mean node confidence
        - graph_data["urgent_nodes"]             — list of nodes below 85 % conf
        """
        nodes: List[Dict] = graph_data.get("nodes", [])
        edges: List[Dict] = graph_data.get("edges", [])

        if not nodes:
            logger.info("AmbiguityDetector: no nodes to score — skipping.")
            graph_data["overall_graph_confidence"] = 100.0
            graph_data["urgent_nodes"] = []
            return graph_data

        logger.info(f"AmbiguityDetector: scoring {len(nodes)} nodes …")

        # ── Pre-compute edge lookup maps ──────────────────────────────────────
        # adjacent_edges[node_id] = list of edge dicts touching that node
        adjacent_edges: Dict[str, List[Dict]] = {n["id"]: [] for n in nodes}
        for edge in edges:
            src = edge.get("source")
            tgt = edge.get("target")
            if src in adjacent_edges:
                adjacent_edges[src].append(edge)
            if tgt and tgt in adjacent_edges:
                adjacent_edges[tgt].append(edge)

        # ── Score each node ───────────────────────────────────────────────────
        urgent_nodes: List[Dict] = []

        for node in nodes:
            node_id = node["id"]
            penalty = 0.0

            adj = adjacent_edges.get(node_id, [])

            # 1. Low edge confidence (< 60 on 0-100 scale or < 0.60 on 0-1 scale)
            has_low_conf = False
            for e in adj:
                c = e.get("confidence", 75.0)
                if c is not None and (c < 60.0 if c > 1.0 else c < 0.60):
                    has_low_conf = True
                    break
            if has_low_conf:
                penalty += _PENALTY_LOW_EDGE_CONF

            # 2. Hidden / inferred node
            if node.get("is_hidden", False):
                penalty += _PENALTY_HIDDEN_NODE

            # 3. Contradicted edges touching this node
            if any(e.get("is_contradicted", False) for e in adj):
                penalty += _PENALTY_CONTRADICTED_EDGE

            # 4. Bad or missing GP fit (DATA path)
            fit = node.get("fit_metrics", {})
            if fit:
                fit_status = fit.get("status", "")
                if fit_status not in ("FITTED", "SOURCE"):
                    penalty += _PENALTY_BAD_FIT

            # 5. Structural isolation (degree == 1)
            if len(adj) == 1:
                penalty += _PENALTY_LOW_DEGREE

            ambiguity = min(penalty, 100.0)
            confidence = round(100.0 - ambiguity, 2)

            node["ambiguity_score"] = round(ambiguity, 2)
            node["urgency_score"] = round(ambiguity, 2)  # mirrored for ConfidenceChecker
            node["confidence"] = confidence

            logger.debug(
                f"  {node_id}: ambiguity={ambiguity:.1f}  confidence={confidence:.1f}"
            )

            if confidence < 85.0:
                urgent_nodes.append({
                    "node_id": node_id,
                    "urgency_score": ambiguity,
                    "confidence": confidence,
                })

        # ── Graph-level confidence ────────────────────────────────────────────
        all_conf = [n.get("confidence", 100.0) for n in nodes]
        overall = round(sum(all_conf) / len(all_conf), 2)

        graph_data["overall_graph_confidence"] = overall
        graph_data["urgent_nodes"] = sorted(
            urgent_nodes, key=lambda x: x["urgency_score"], reverse=True
        )

        logger.info(
            f"AmbiguityDetector complete: overall_confidence={overall:.1f}%, "
            f"urgent_nodes={len(urgent_nodes)}"
        )
        return graph_data