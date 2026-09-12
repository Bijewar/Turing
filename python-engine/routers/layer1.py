"""
routers/layer1.py — Layer 1 Data Ingestion Endpoints

Exposes both the high-level orchestration endpoints (upload, status)
and the granular, step-by-step pipeline microservices for the Data Path and Text Path.
"""
import os
import re
import tempfile
import asyncio
import logging
from typing import Dict, Any, Optional, List, Union

import pandas as pd
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, Form, BackgroundTasks
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

# Mayank's DB & Orchestration Imports
from database.database import get_db
from database import crud
from schemas.run import RunStatus

# Layer 1 Detection & Pipeline Imports
from services.layer1.file_detector import detect_input_type, validate_file_exists, InputType, UniversalFileDetector
from services.layer1.extractor import UniversalExtractor, AmbiguityDetector
from services.layer1.pc_algorithm import PCGraphBuilder
from services.layer1.ontology_builder import LLMGraphBuilder
from services.layer1.validator import GraphValidator, ConfidenceChecker
from services.layer1.gaussian_process import StructuralFitter
from services.layer1.classifier import NodeClassifier

logger = logging.getLogger(__name__)
router = APIRouter()

# ==============================================================================
# Pydantic Request Models (BUG #28)
# ==============================================================================

class BuildGraphRequest(BaseModel):
    path_type: str = Field(..., description="'DATA' or 'TEXT'")
    data: Optional[List[Dict[str, Any]]] = Field(default=None, description="Array of row objects for DATA path")
    text: Optional[str] = Field(default=None, description="Text string for TEXT path")
    model_name: Optional[str] = Field(default=None, description="Optional LLM model name")

class ValidateGraphRequest(BaseModel):
    graph: Dict[str, Any] = Field(..., description="Causal graph object containing 'nodes' and 'edges'")
    model_name: Optional[str] = Field(default=None, description="Optional LLM model name")

class FitEquationsRequest(BaseModel):
    graph: Dict[str, Any] = Field(..., description="Causal graph object")
    path_type: str = Field(default="DATA", description="'DATA' or 'TEXT'")
    data: Optional[List[Dict[str, Any]]] = Field(default=None, description="Array of row objects for DATA path fitting")

class ClassifyNodesRequest(BaseModel):
    graph: Dict[str, Any] = Field(..., description="Causal graph object containing 'nodes' and 'edges'")
    model_name: Optional[str] = Field(default=None, description="Optional LLM model name")

class AnalyzeAmbiguityRequest(BaseModel):
    graph: Dict[str, Any] = Field(..., description="Causal graph object containing 'nodes' and 'edges'")

class EvaluateConfidenceRequest(BaseModel):
    graph: Dict[str, Any] = Field(..., description="Causal graph object containing 'nodes' and 'edges'")
    path_type: str = Field(default="DATA", description="'DATA' or 'TEXT'")


# Helper to stream file in chunks and enforce size limits (BUG #3 / SWARM-8)
async def _read_file_safe(file: UploadFile, max_mb: int = 50) -> bytes:
    max_bytes = max_mb * 1024 * 1024
    chunks = []
    total = 0
    chunk_size = 1024 * 1024  # 1MB
    while True:
        chunk = await file.read(chunk_size)
        if not chunk:
            break
        total += len(chunk)
        if total > max_bytes:
            raise HTTPException(status_code=413, detail=f"File exceeds maximum allowed size of {max_mb}MB.")
        chunks.append(chunk)
    return b"".join(chunks)


async def _process_uploaded_file(run_id: str, tmp_path: str):
    """Background processing task for uploaded files."""
    try:
        logger.info(f"Processing run {run_id} in background from {tmp_path}")
    finally:
        if os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except Exception as e:
                logger.warning(f"Failed to remove temp file {tmp_path}: {e}")


# ==============================================================================
# 1. ORCHESTRATION ENDPOINTS (Mayank)
# ==============================================================================

@router.post(
    "/upload",
    summary="Upload a CSV or document file to start a new analysis run",
    status_code=202,
)
async def upload_file(
    run_id: str,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
):
    """
    Accepts a file upload, persists it temporarily, detects its type,
    and starts the background processing task.
    """
    run = await crud.get_run(db, run_id)
    if not run:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")

    suffix = os.path.splitext(file.filename or "upload")[1] or ".bin"
    tmp_path = ""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp_path = tmp.name
            chunk_size = 1024 * 1024
            total = 0
            max_bytes = 50 * 1024 * 1024
            while True:
                chunk = await file.read(chunk_size)
                if not chunk:
                    break
                total += len(chunk)
                if total > max_bytes:
                    raise HTTPException(status_code=413, detail="File exceeds 50MB limit.")
                tmp.write(chunk)

        if not validate_file_exists(tmp_path):
            raise HTTPException(status_code=400, detail="Uploaded file is empty or unreadable")

        input_type = detect_input_type(tmp_path)
        if input_type == InputType.UNKNOWN:
            raise HTTPException(
                status_code=415,
                detail=f"Unsupported file type: '{suffix}'. Use CSV, PDF, TXT, DOCX, or XLSX."
            )

        await crud.update_run_status(db, run_id, RunStatus.RUNNING)
        background_tasks.add_task(_process_uploaded_file, run_id, tmp_path)

        return {
            "run_id": run_id,
            "filename": file.filename,
            "detected_type": input_type.value,
            "message": "File accepted. Processing pipeline has started.",
        }
    except Exception as e:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
        if isinstance(e, HTTPException):
            raise e
        logger.error(f"Failed in upload_file: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error during upload.")


@router.get(
    "/status/{run_id}",
    summary="Check the processing status of a Layer 1 run"
)
async def get_status(run_id: str, db: AsyncSession = Depends(get_db)):
    """Lightweight status check — returns the current run status and graph if ready."""
    run = await crud.get_run(db, run_id)
    if not run:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")
    return {
        "run_id": run.id,
        "status": run.status,
        "causal_graph": run.get_causal_graph(),
    }


# ==============================================================================
# 2. GRANULAR PIPELINE ENDPOINTS (Harsh)
# ==============================================================================

@router.post("/detect", response_model=Dict[str, Any])
async def detect_file_type(file: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Step 1: Universal File Detector.
    Receives an uploaded file, analyzes its contents,
    and routes it to either the DATA PATH or TEXT PATH.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
        
    try:
        file_bytes = await _read_file_safe(file)
        result = await asyncio.to_thread(UniversalFileDetector.analyze_file, file_bytes, file.filename)
        
        result["filename"] = file.filename
        result["size_bytes"] = len(file_bytes)
        
        return result
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in detect_file_type: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal error during file detection.")


@router.post("/extract", response_model=Dict[str, Any])
async def extract_data(
    path_type: str = Form(...),
    file: UploadFile = File(...)
) -> Dict[str, Any]:
    """
    Step 2: Universal Data Extractor.
    Takes a file and the detected path_type ('DATA' or 'TEXT').
    Returns a standardized mathematical table (JSON array) or a clean text string.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
        
    if path_type not in ["DATA", "TEXT"]:
        raise HTTPException(status_code=400, detail="path_type must be DATA or TEXT")
        
    try:
        file_bytes = await _read_file_safe(file)
        
        response = {
            "path_type": path_type,
            "filename": file.filename,
            "data": None,
            "text": None,
            "warnings": [],
            "columns": []
        }
        
        if path_type == "DATA":
            df, warnings = await asyncio.to_thread(UniversalExtractor.extract_data, file_bytes, file.filename)
            response["data"] = df.to_dict(orient="records")
            response["columns"] = df.columns.tolist()
            response["warnings"] = warnings
        else:
            text = await asyncio.to_thread(UniversalExtractor.extract_text, file_bytes, file.filename)
            response["text"] = text
            
        return response
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in extract_data: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal error during data extraction.")


@router.post("/build-graph", response_model=Dict[str, Any])
async def build_graph(
    payload: BuildGraphRequest
) -> Dict[str, Any]:
    """
    Step 3: Graph Builder.
    Takes the extracted payload (data or text) and returns a unified 
    NetworkX causal graph representation.
    """
    path_type = payload.path_type
    
    if path_type not in ["DATA", "TEXT"]:
        raise HTTPException(status_code=400, detail="Invalid or missing path_type")
        
    try:
        if path_type == "DATA":
            data = payload.data
            if not data:
                raise ValueError("DATA path requires 'data' array in payload")
            
            df = pd.DataFrame(data)
            graph_json = await asyncio.to_thread(PCGraphBuilder.build_graph, df)
            return graph_json
            
        else:
            text = payload.text
            model_name = payload.model_name or os.getenv("DEFAULT_LLM_MODEL", "gpt-4o")
            
            if not text:
                raise ValueError("TEXT path requires 'text' string in payload")
                
            graph_json = await asyncio.to_thread(LLMGraphBuilder.build_graph, text, model_name)
            return graph_json
            
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Error in build_graph: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal error during causal graph building.")


@router.post("/validate-graph", response_model=Dict[str, Any])
async def validate_graph(
    payload: ValidateGraphRequest
) -> Dict[str, Any]:
    """
    Step 4: Graph Validator.
    Takes a raw JSON graph {"nodes": [], "edges": []} from Step 3,
    runs the autonomous physics checks, breaks cycles, and flags contradictions.
    """
    graph_data = payload.graph
    model_name = payload.model_name or os.getenv("DEFAULT_LLM_MODEL", "gpt-4o")
    
    if not graph_data or not isinstance(graph_data, dict):
        raise HTTPException(status_code=400, detail="Missing or invalid 'graph' object in payload")
        
    if "nodes" not in graph_data or "edges" not in graph_data:
        raise HTTPException(status_code=400, detail="'graph' must contain 'nodes' and 'edges' arrays")
        
    try:
        validated_result = await asyncio.to_thread(GraphValidator.validate, graph_data, model_name)
        return validated_result
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Error in validate_graph: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal error during graph validation.")


@router.post("/fit-equations", response_model=Dict[str, Any])
async def fit_equations(
    payload: FitEquationsRequest
) -> Dict[str, Any]:
    """
    Step 5: Structural Equation Fitter.
    Fits Gaussian Processes to the edges (DATA path only), saves the models,
    and returns fit metrics.
    """
    graph_data = payload.graph
    path_type = payload.path_type
    data = payload.data or []
    
    if not graph_data or not isinstance(graph_data, dict):
        raise HTTPException(status_code=400, detail="Missing or invalid 'graph' object in payload")
        
    if not path_type or path_type not in ["DATA", "TEXT"]:
        raise HTTPException(status_code=400, detail="path_type must be DATA or TEXT")
        
    try:
        if path_type == "DATA":
            if not data:
                raise ValueError("DATA path requires 'data' array in payload for fitting")
            df = pd.DataFrame(data)
            fitted_graph = await asyncio.to_thread(StructuralFitter.fit_graph, df, graph_data, path_type)
        else:
            fitted_graph = await asyncio.to_thread(StructuralFitter.fit_graph, pd.DataFrame(), graph_data, path_type)
            
        return fitted_graph
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Error in fit_equations: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal error during equation fitting.")


@router.post("/classify-nodes", response_model=Dict[str, Any])
async def classify_nodes(
    payload: ClassifyNodesRequest
) -> Dict[str, Any]:
    """
    Step 6: Node Classifier.
    Takes a graph from Step 5, calculates in/out degrees, and applies
    semantic labels and UI themes (Source, Sink, Bottleneck, Mediator).
    """
    graph_data = payload.graph
    
    if not graph_data or not isinstance(graph_data, dict):
        raise HTTPException(status_code=400, detail="Missing or invalid 'graph' object in payload")
        
    if "nodes" not in graph_data or "edges" not in graph_data:
        raise HTTPException(status_code=400, detail="'graph' must contain 'nodes' and 'edges' arrays")
        
    try:
        classified_graph = await asyncio.to_thread(NodeClassifier.classify_graph, graph_data)
        return classified_graph
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Error in classify_nodes: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal error during node classification.")


@router.post("/detect-ambiguity", response_model=Dict[str, Any])
@router.post("/analyze-ambiguity", response_model=Dict[str, Any])
async def analyze_ambiguity(
    payload: AnalyzeAmbiguityRequest
) -> Dict[str, Any]:
    """
    Step 7: Ambiguity Detector.
    Takes the classified graph, scores edges, and ranks unknown nodes.
    """
    graph_data = payload.graph
    
    if not graph_data or not isinstance(graph_data, dict):
        raise HTTPException(status_code=400, detail="Missing or invalid 'graph' object in payload")
        
    try:
        final_graph = await asyncio.to_thread(AmbiguityDetector.analyze_graph, graph_data)
        return final_graph
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Error in analyze_ambiguity: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal error during ambiguity detection.")


@router.post("/confidence-check", response_model=Dict[str, Any])
@router.post("/evaluate-confidence", response_model=Dict[str, Any])
async def evaluate_confidence(
    payload: EvaluateConfidenceRequest
) -> Dict[str, Any]:
    """
    Step 8: Confidence Check.
    Makes the final routing decision and assembles the Output Package.
    """
    graph_data = payload.graph
    path_type = payload.path_type
    
    if not graph_data or not isinstance(graph_data, dict):
        raise HTTPException(status_code=400, detail="Missing or invalid 'graph' object in payload")
        
    try:
        output_package = await asyncio.to_thread(ConfidenceChecker.evaluate_graph, graph_data, path_type)
        return output_package
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Error in evaluate_confidence: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal error during confidence evaluation.")