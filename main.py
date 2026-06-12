import os
import time
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from config.settings import settings, BASE_DIR
from utils.logger import get_logger
from data_engine import (
    load_csv,
    validate_dataset,
    extract_schema_metadata,
    parse_natural_language_query,
    validate_filter_payload,
    apply_filters,
    generate_analytics,
    generate_executive_insights,
    generate_executive_report_pdf,
    verify_reporting_pipeline,
    get_deterministic_graph_recommendations,
    profile_dataset,
    validate_math_expression,
    evaluate_math_expression,
    get_next_api_key
)

logger = get_logger("fastapi_app")

# Initialize FastAPI App
app = FastAPI(
    title="NGO Data Analytics API",
    description="REST API backend for the NGO Data Analytics Platform",
    version="1.0.0"
)

# Enable CORS for local dev and production frontends (e.g. Vercel)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this to specific Vercel domains in production if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Helpers for file handling
UPLOAD_DIR = Path(BASE_DIR) / "data"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
UPLOADED_FILE_PATH = UPLOAD_DIR / "uploaded_dataset.csv"

def get_active_dataset_path() -> Path:
    """Returns the path to the uploaded dataset if exists, otherwise defaults."""
    if UPLOADED_FILE_PATH.exists():
        return UPLOADED_FILE_PATH
    return Path(settings.DATASET_PATH)

def get_active_dataset() -> pd.DataFrame:
    """Loads the active dataset and validates it."""
    path = get_active_dataset_path()
    if not path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No dataset available. Please upload a CSV file first."
        )
    try:
        df = load_csv(path)
        if not validate_dataset(df):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Dataset validation failed."
            )
        return df
    except Exception as e:
        logger.exception("Error loading active dataset:")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to load dataset: {str(e)}"
        )

def get_distribution_data(df: pd.DataFrame, column: str) -> List[Dict[str, Any]]:
    """Generates histogram/distribution data for client-side rendering (JSON)."""
    if df.empty or column not in df.columns:
        return []
    
    series = df[column].dropna()
    if series.empty:
        return []
        
    try:
        # Numeric column distribution
        if pd.api.types.is_numeric_dtype(series):
            counts, bins = np.histogram(series, bins=10)
            distribution = []
            for i in range(len(counts)):
                bucket_label = f"{bins[i]:.1f} - {bins[i+1]:.1f}"
                distribution.append({
                    "bucket": bucket_label,
                    "count": int(counts[i])
                })
            return distribution
        else:
            # Categorical column distribution
            counts = series.value_counts().head(10)
            return [
                {"bucket": str(k), "count": int(v)} 
                for k, v in counts.items()
            ]
    except Exception as e:
        logger.error(f"Error calculating distribution for column {column}: {e}")
        return []

def save_distribution_plot_image(df: pd.DataFrame, column: str) -> Path:
    """Generates a Seaborn distribution plot and saves to disk for PDF report."""
    chart_path = Path(BASE_DIR) / "reports" / "active_chart.png"
    chart_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        if df.empty or column not in df.columns:
            # Save an empty chart or simple visual indicator
            fig, ax = plt.subplots(figsize=(7, 3.5), facecolor="#f8fafc")
            ax.text(0.5, 0.5, "No data available", ha='center', va='center')
            plt.tight_layout()
            fig.savefig(chart_path, dpi=150)
            plt.close(fig)
            return chart_path

        fig, ax = plt.subplots(figsize=(7, 3.5), facecolor="#f8fafc")
        ax.set_facecolor("#ffffff")
        
        sns.histplot(
            data=df, 
            x=column, 
            kde=True, 
            color="#6366f1", 
            ax=ax, 
            bins=15, 
            alpha=0.6,
            edgecolor="white",
            line_kws={"linewidth": 2.5}
        )
        
        ax.set_title(f"Distribution of {column}", fontsize=11, fontweight="bold", pad=12, color="#0f172a")
        ax.set_xlabel(column, fontsize=8.5, color="#475569", labelpad=8)
        ax.set_ylabel("Count", fontsize=8.5, color="#475569", labelpad=8)
        ax.tick_params(labelsize=8, colors="#64748b")
        ax.grid(True, linestyle="--", alpha=0.5, color="#e2e8f0")
        
        for spine in ["top", "right", "left", "bottom"]:
            ax.spines[spine].set_visible(False)
            
        plt.tight_layout()
        fig.savefig(chart_path, dpi=150, facecolor=fig.get_facecolor(), edgecolor="none", bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        logger.error(f"Error saving distribution plot: {e}", exc_info=True)
    return chart_path

# ==============================================================================
# MODELS
# ==============================================================================

class FilterItem(BaseModel):
    column: str
    operator: str
    value: Any

class QueryRequest(BaseModel):
    filters: Optional[List[FilterItem]] = Field(default_factory=list)
    natural_language_query: Optional[str] = None
    math_expression: Optional[str] = None
    chart_column: Optional[str] = None
    api_key_override: Optional[str] = None

class InsightsRequest(BaseModel):
    analytics: Dict[str, Any]
    api_key_override: Optional[str] = None

class ReportRequest(BaseModel):
    filters: Dict[str, Any]
    analytics: Dict[str, Any]
    insights: List[str]
    chart_column: Optional[str] = None

# ==============================================================================
# ENDPOINTS
# ==============================================================================

@app.get("/health")
def health_check():
    """Simple healthcheck endpoint."""
    dataset_path = get_active_dataset_path()
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "active_dataset": dataset_path.name if dataset_path.exists() else "None"
    }

@app.post("/api/upload")
async def upload_dataset(file: UploadFile = File(...)):
    """Uploads a CSV dataset, stores it, and extracts metadata schema."""
    if not file.filename.endswith(".csv"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only CSV files are supported."
        )
        
    try:
        contents = await file.read()
        
        # Validate CSV contents can be parsed before saving
        import io
        try:
            pd.read_csv(io.BytesIO(contents), nrows=5)
        except Exception as parse_err:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Uploaded file is not a valid CSV format: {parse_err}"
            )
            
        # Write bytes
        UPLOADED_FILE_PATH.write_bytes(contents)
        logger.info(f"Received and saved dataset file: {file.filename}")
        
        # Load dataset & extract metadata
        df = get_active_dataset()
        metadata = extract_schema_metadata(df)
        
        # Generate hash
        dataset_hash = hashlib.md5(contents).hexdigest()
        
        return {
            "success": True,
            "filename": file.filename,
            "hash": dataset_hash,
            "schema": metadata
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Upload process failed:")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred during upload: {str(e)}"
        )

@app.get("/api/schema")
def get_schema():
    """Extracts schema metadata details for the active dataset."""
    df = get_active_dataset()
    metadata = extract_schema_metadata(df)
    return {
        "success": True,
        "schema": metadata
    }

@app.post("/api/query")
def run_query(request: QueryRequest):
    """Parses queries, filters dataset, runs analytics, and calculates custom metrics."""
    df_original = get_active_dataset()
    metadata = extract_schema_metadata(df_original)
    
    # 1. Profile dataset (caching enabled)
    dataset_path = get_active_dataset_path()
    try:
        file_hash = hashlib.md5(dataset_path.read_bytes()).hexdigest()
    except Exception:
        file_hash = "default"
        
    dataset_profile = profile_dataset(df_original, file_hash, api_key=request.api_key_override)
    
    # 2. Process Natural Language Query via LLM
    inferred_filters = []
    inferred_expression = None
    
    if request.natural_language_query and request.natural_language_query.strip():
        try:
            nl_payload = parse_natural_language_query(
                request.natural_language_query,
                metadata,
                api_key=request.api_key_override,
                dataset_profile=dataset_profile
            )
            if "error" in nl_payload:
                logger.error(f"NL parser returned error: {nl_payload['error']}")
            else:
                inferred_filters = nl_payload.get("filters", [])
                inferred_expression = nl_payload.get("expression")
        except Exception as parse_err:
            logger.exception("NL parsing execution failed:")
            
    # 3. Merge filters (LLM inferred filters + Manual explicit filters)
    client_filters = [f.dict() for f in request.filters] if request.filters else []
    combined_filters = inferred_filters + client_filters
    combined_payload = {"filters": combined_filters}
    
    # 4. Filter validation and execution
    if not validate_filter_payload(combined_payload, df_original):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Security validation failed for the combined filter criteria."
        )
        
    filtered_df = apply_filters(df_original, combined_payload)
    
    # 5. Evaluate custom math expression (prefer user specified, fall back to inferred)
    metric_value = None
    active_expression = request.math_expression or inferred_expression
    
    if active_expression and active_expression.strip():
        if validate_math_expression(active_expression, list(df_original.columns)):
            try:
                metric_value = evaluate_math_expression(filtered_df, active_expression)
            except Exception as eval_err:
                logger.error(f"Math evaluation failed: {eval_err}")
                metric_value = f"Error: {str(eval_err)}"
        else:
            metric_value = "Error: Blocked (Failed safety check)"
            
    # 6. Generate analytics
    analytics = generate_analytics(filtered_df)
    
    # 7. Chart Column selection and bucket aggregation
    # Default chart column: first numeric column from metadata
    numeric_cols = [c for c, m in metadata.items() if m.get("dtype", "").startswith(("int", "float"))]
    selected_chart_col = request.chart_column
    if not selected_chart_col or selected_chart_col not in df_original.columns:
        selected_chart_col = numeric_cols[0] if numeric_cols else list(df_original.columns)[0]
        
    chart_data = get_distribution_data(filtered_df, selected_chart_col)
    
    # 8. Graph recommendations
    graph_recs = get_deterministic_graph_recommendations(metadata)
    
    return {
        "success": True,
        "matched_count": len(filtered_df),
        "total_count": len(df_original),
        "metric_value": metric_value,
        "metric_expression": active_expression,
        "filters_applied": combined_filters,
        "analytics": analytics,
        "chart_column": selected_chart_col,
        "chart_data": chart_data,
        "recommendations": graph_recs
    }

@app.post("/api/insights")
def get_insights(request: InsightsRequest):
    """Runs LLM analyst narrative report on numerical aggregates."""
    try:
        insights = generate_executive_insights(
            request.analytics,
            api_key=request.api_key_override
        )
        return {
            "success": True,
            "insights": insights
        }
    except Exception as e:
        logger.exception("Insights API endpoint failed:")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Narrative generation failed: {str(e)}"
        )

@app.post("/api/report")
def export_report(request: ReportRequest):
    """Compiles a PDF executive summary and returns it for download."""
    df_original = get_active_dataset()
    
    # Filter dataset based on the request's filters payload
    try:
        if not validate_filter_payload(request.filters, df_original):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Security check failed for pdf report filters."
            )
        filtered_df = apply_filters(df_original, request.filters)
    except Exception as f_err:
        logger.exception("Filtering for report compilation failed:")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Report filter failed: {str(f_err)}"
        )
        
    # Generate and save chart image locally
    chart_col = request.chart_column
    if not chart_col or chart_col not in df_original.columns:
        # Fall back to first numerical column
        metadata = extract_schema_metadata(df_original)
        numeric_cols = [c for c, m in metadata.items() if m.get("dtype", "").startswith(("int", "float"))]
        chart_col = numeric_cols[0] if numeric_cols else list(df_original.columns)[0]
        
    chart_file_path = save_distribution_plot_image(filtered_df, chart_col)
    
    try:
        # Generate the PDF file content (bytes)
        pdf_data = generate_executive_report_pdf(
            filters=request.filters,
            analytics_summary=request.analytics,
            insights=request.insights,
            chart_path=str(chart_file_path)
        )
        
        # Verify structure
        check_results = verify_reporting_pipeline()
        if not check_results.get("download_ready", False):
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Generated report failed structural verification checks."
            )
            
        # Write to temporary file and stream back
        temp_pdf_path = Path(BASE_DIR) / "reports" / "temp_download.pdf"
        temp_pdf_path.write_bytes(pdf_data)
        
        def iterfile():
            with open(temp_pdf_path, mode="rb") as f:
                yield from f
                
        return StreamingResponse(
            iterfile(),
            media_type="application/pdf",
            headers={
                "Content-Disposition": f"attachment; filename=NGO_Executive_Report_{int(time.time())}.pdf"
            }
        )
    except Exception as pdf_err:
        logger.exception("PDF pipeline compilation failed:")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to compile PDF document: {str(pdf_err)}"
        )
