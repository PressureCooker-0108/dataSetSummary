import os
import time
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import hashlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from config.settings import settings, BASE_DIR
from utils.logger import get_logger
from data_engine import (
    load_csv,
    validate_dataset,
    analyze_schema,
    generate_summary_report,
    extract_schema_metadata,
    parse_natural_language_query,
    validate_filter_payload,
    apply_filters,
    generate_analytics,
    generate_executive_insights,
    generate_executive_report_pdf,
    verify_reporting_pipeline,
    verify_openrouter_connection,
    get_deterministic_graph_recommendations
)

def check_any_key_configured(override_keys_str: str = None) -> bool:
    """Checks if any OpenRouter API Key is configured in settings, env, or overrides."""
    if override_keys_str and override_keys_str.strip():
        return True
    for key in [settings.OPENROUTER_API_KEY, settings.OPENROUTER_API_KEY_1, 
                settings.OPENROUTER_API_KEY_2, settings.OPENROUTER_API_KEY_3, 
                settings.OPENROUTER_API_KEY_4]:
        if key:
            return True
    for env_name in ["OPENROUTER_API_KEY", "OPENROUTER_API_KEY_1", "OPENROUTER_API_KEY_2", 
                     "OPENROUTER_API_KEY_3", "OPENROUTER_API_KEY_4"]:
        if os.getenv(env_name):
            return True
    return False


@st.cache_data(show_spinner="Verifying API connection...", ttl=600)
def check_api_connection(api_keys_str: str) -> bool:
    """Performs a low-token connection check using the rotated API key."""
    from data_engine import get_next_api_key
    key = get_next_api_key(api_keys_str)
    if not key:
        return False
    return verify_openrouter_connection(key)

# Initialize application logger
logger = get_logger("streamlit_app")

# Page Configuration
st.set_page_config(
    page_title="NGO Data Analytics Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Premium Custom CSS Injection for modern layout and typography
def inject_custom_styles() -> None:
    st.markdown("""
    <style>
        /* Import premium fonts */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@400;600;800&display=swap');
        
        html, body {
            font-family: 'Inter', sans-serif;
        }
        
        /* Enforce page light background */
        [data-testid="stAppViewContainer"] {
            background-color: #f8fafc !important;
        }
        
        /* Force dark text color across all main container text elements for high readability */
        [data-testid="stAppViewContainer"] p, 
        [data-testid="stAppViewContainer"] span, 
        [data-testid="stAppViewContainer"] label, 
        [data-testid="stAppViewContainer"] li, 
        [data-testid="stAppViewContainer"] h1, 
        [data-testid="stAppViewContainer"] h2, 
        [data-testid="stAppViewContainer"] h3, 
        [data-testid="stAppViewContainer"] h4, 
        [data-testid="stAppViewContainer"] h5, 
        [data-testid="stAppViewContainer"] h6,
        [data-testid="stAppViewContainer"] div {
            color: #0f172a;
        }
        
        /* Gradient titles */
        .main-header {
            font-family: 'Outfit', sans-serif;
            font-size: 2.8rem;
            font-weight: 800;
            background: linear-gradient(135deg, #6366f1 0%, #0d9488 50%, #0284c7 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent !important;
            margin-bottom: 0.1rem;
        }
        
        .sub-header {
            font-size: 1.1rem;
            color: #64748b !important;
            margin-bottom: 2rem;
        }
        
        /* Premium Card Layouts with glassmorphic look, gradient shadows and hover effect */
        .kpi-card {
            background: #ffffff !important;
            border-radius: 16px;
            padding: 1.5rem 1rem;
            box-shadow: 0 4px 20px 0 rgba(148, 163, 184, 0.08);
            border: 1px solid rgba(226, 232, 240, 0.8);
            transition: all 0.4s cubic-bezier(0.16, 1, 0.3, 1);
            text-align: center;
            position: relative;
            overflow: hidden;
        }
        
        .kpi-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 4px;
            background: linear-gradient(90deg, #6366f1, #0d9488);
            opacity: 0.8;
        }
        
        .kpi-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 12px 30px 0 rgba(99, 102, 241, 0.12);
            border-color: rgba(99, 102, 241, 0.4);
        }
        
        .kpi-label {
            font-size: 0.75rem;
            font-weight: 600;
            color: #64748b !important;
            text-transform: uppercase;
            letter-spacing: 0.075em;
            margin-bottom: 0.5rem;
        }
        
        .kpi-value {
            font-family: 'Outfit', sans-serif;
            font-size: 1.8rem;
            font-weight: 700;
            color: #0f172a !important;
        }
        
        /* Status Badges */
        .badge {
            display: inline-flex;
            align-items: center;
            padding: 0.35rem 0.85rem;
            border-radius: 9999px;
            font-size: 0.8rem;
            font-weight: 600;
            margin-bottom: 0.5rem;
        }
        
        .badge-success {
            background-color: #ecfdf5 !important;
            color: #047857 !important;
            border: 1px solid #a7f3d0 !important;
        }
        
        .badge-error {
            background-color: #fff5f5 !important;
            color: #c53030 !important;
            border: 1px solid #feb2b2 !important;
        }
        
        .badge-warning {
            background-color: #fffdf5 !important;
            color: #d97706 !important;
            border: 1px solid #fde68a !important;
        }

        /* Streamlit primary and secondary buttons style override */
        .stButton > button {
            border-radius: 10px !important;
            font-family: 'Inter', sans-serif !important;
            font-weight: 600 !important;
            font-size: 0.9rem !important;
            padding: 0.5rem 1.25rem !important;
            transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1) !important;
        }
        
        .stButton > button:not([kind="primary"]) {
            border: 1px solid #e2e8f0 !important;
            color: #334155 !important;
            background-color: #ffffff !important;
        }
        
        .stButton > button:not([kind="primary"]):hover {
            border-color: #6366f1 !important;
            color: #6366f1 !important;
            background-color: #f8fafc !important;
            box-shadow: 0 4px 12px rgba(99, 102, 241, 0.08) !important;
            transform: translateY(-1px);
        }
        
        .stButton > button[kind="primary"] {
            background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%) !important;
            color: white !important;
            border: none !important;
            box-shadow: 0 4px 14px rgba(99, 102, 241, 0.2) !important;
        }
        
        .stButton > button[kind="primary"]:hover {
            background: linear-gradient(135deg, #4f46e5 0%, #4338ca 100%) !important;
            box-shadow: 0 6px 20px rgba(99, 102, 241, 0.3) !important;
            transform: translateY(-1px);
        }

        /* Style text inputs */
        .stTextInput input, .stTextArea textarea {
            border-radius: 10px !important;
            border: 1px solid #e2e8f0 !important;
            font-family: 'Inter', sans-serif !important;
            padding: 0.6rem 1rem !important;
            background-color: #ffffff !important;
            color: #0f172a !important;
            transition: all 0.2s ease-in-out !important;
            box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05) !important;
        }
        
        .stTextInput input:focus, .stTextArea textarea:focus {
            border-color: #6366f1 !important;
            box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.15) !important;
        }

        /* Style selectbox & multiselect closed state dropdown boxes */
        .stSelectbox div[data-baseweb="select"], 
        .stMultiSelect div[data-baseweb="select"] {
            background-color: #ffffff !important;
            border-radius: 10px !important;
            border: 1px solid #e2e8f0 !important;
        }
        
        .stSelectbox div[data-baseweb="select"] *, 
        .stMultiSelect div[data-baseweb="select"] * {
            color: #0f172a !important;
        }

        /* Force dark text inside dataframes and tables */
        .stDataFrame div {
            color: #0f172a !important;
        }

        /* Sidebar layout styling */
        [data-testid="stSidebar"] {
            background-color: #f8fafc !important;
            border-right: 1px solid #edf2f7 !important;
        }
        
        [data-testid="stSidebar"] h2 {
            border-bottom: 2px solid #e2e8f0;
            padding-bottom: 0.5rem;
            margin-bottom: 1.5rem;
            color: #0f172a !important;
        }

        /* Force sidebar labels, headers, and text to be dark slate */
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] span,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3,
        [data-testid="stSidebar"] h4,
        [data-testid="stSidebar"] h5,
        [data-testid="stSidebar"] h6 {
            color: #1e293b !important;
        }

        /* Modern tabs styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 12px !important;
            background-color: #f1f5f9 !important;
            padding: 6px !important;
            border-radius: 14px !important;
            border-bottom: none !important;
        }
        
        .stTabs [data-baseweb="tab"] {
            border-radius: 10px !important;
            padding: 10px 20px !important;
            font-weight: 600 !important;
            font-size: 0.95rem !important;
            color: #64748b !important;
            background-color: transparent !important;
            transition: all 0.25s ease !important;
            border-bottom: none !important;
        }
        
        .stTabs [data-baseweb="tab"]:hover {
            color: #4f46e5 !important;
        }
        
        .stTabs [aria-selected="true"] {
            background-color: #ffffff !important;
            color: #4f46e5 !important;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05) !important;
        }

        /* Card panels for instructions */
        .welcome-card {
            background-color: #ffffff !important;
            border-radius: 16px;
            padding: 2.5rem;
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.05), 0 8px 10px -6px rgba(0, 0, 0, 0.05);
            border: 1px solid #e2e8f0;
            margin-top: 1rem;
        }
        
        .step-badge {
            background: linear-gradient(135deg, #6366f1, #0d9488);
            color: white !important;
            border-radius: 50%;
            width: 28px;
            height: 28px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            font-weight: bold;
            margin-right: 0.75rem;
            font-size: 0.9rem;
            flex-shrink: 0;
        }

        .step-container {
            display: flex;
            align-items: flex-start;
            margin-bottom: 1.25rem;
        }

        .step-text {
            font-size: 1rem;
            color: #334155 !important;
            line-height: 1.6;
        }
    </style>
    """, unsafe_allow_html=True)


inject_custom_styles()


# ==============================================================================
# SESSION STATE INITIALIZATION
# ==============================================================================

def initialize_session_state(metadata: Dict[str, Any]) -> None:
    """Initializes all Streamlit session state keys with robust defaults."""
    if "state_initialized" not in st.session_state:
        logger.info("Initializing application Session State variables.")
        st.session_state.state_initialized = True
        st.session_state.nl_query = ""
        st.session_state.nl_payload = {"filters": []}
        st.session_state.selected_chart_col = "Age"
        st.session_state.selected_groupby_col = "Gender"
        st.session_state.api_key_override = ""
        
        # Default numeric range configurations with safety guards
        range_age = metadata.get("Age", {}).get("range", {}) if "Age" in metadata else None
        if range_age:
            st.session_state.min_age = int(range_age.get("min", 18))
            st.session_state.max_age = int(range_age.get("max", 60))
        else:
            st.session_state.min_age = 18
            st.session_state.max_age = 60
            
        range_weight = metadata.get("Weight (kg)", {}).get("range", {}) if "Weight (kg)" in metadata else None
        if range_weight:
            st.session_state.min_weight = int(range_weight.get("min", 40))
            st.session_state.max_weight = int(range_weight.get("max", 120))
        else:
            st.session_state.min_weight = 40
            st.session_state.max_weight = 120


# ==============================================================================
# VISUALIZATION ENGINE (MATPLOTLIB & SEABORN)
# ==============================================================================

def render_distribution_plot(dataframe: pd.DataFrame, column_name: str, show_on_streamlit: bool = True) -> None:
    """Renders a Seaborn distribution plot (histogram + KDE) to Streamlit and saves to disk."""
    try:
        if dataframe.empty or column_name not in dataframe.columns:
            if show_on_streamlit:
                st.info("No data available to plot.")
            return

        fig, ax = plt.subplots(figsize=(7, 3.5), facecolor="#f8fafc")
        ax.set_facecolor("#ffffff")
        
        primary_color = "#6366f1"  # Indigo accent color
        
        sns.histplot(
            data=dataframe, 
            x=column_name, 
            kde=True, 
            color=primary_color, 
            ax=ax, 
            bins=15, 
            alpha=0.6,
            edgecolor="white",
            line_kws={"linewidth": 2.5}
        )
        
        ax.set_title(f"Distribution of {column_name}", fontsize=11, fontweight="bold", pad=12, color="#0f172a")
        ax.set_xlabel(column_name, fontsize=8.5, color="#475569", labelpad=8)
        ax.set_ylabel("Count", fontsize=8.5, color="#475569", labelpad=8)
        ax.tick_params(labelsize=8, colors="#64748b")
        
        ax.grid(True, linestyle="--", alpha=0.5, color="#e2e8f0")
        for spine in ["top", "right", "left", "bottom"]:
            ax.spines[spine].set_visible(False)
            
        plt.tight_layout()
        
        # Save chart copy to reports directory for report generator
        try:
            chart_path = Path(BASE_DIR) / "reports" / "active_chart.png"
            chart_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(chart_path, dpi=150, facecolor=fig.get_facecolor(), edgecolor="none", bbox_inches="tight")
        except Exception as save_err:
            logger.error(f"Failed to save active chart: {save_err}")
            
        if show_on_streamlit:
            st.pyplot(fig)
        plt.close(fig)
    except Exception as e:
        logger.error(f"Error rendering distribution plot: {e}", exc_info=True)
        if show_on_streamlit:
            st.error("Failed to generate distribution visualization.")


def render_group_boxplot(dataframe: pd.DataFrame, numeric_col: str, group_col: str, show_on_streamlit: bool = True) -> None:
    """Renders a Seaborn Box Plot grouped by a categorical column and saves to disk."""
    try:
        if dataframe.empty or numeric_col not in dataframe.columns or group_col not in dataframe.columns:
            if show_on_streamlit:
                st.info("No grouping data available.")
            return

        fig, ax = plt.subplots(figsize=(7, 3.5), facecolor="#f8fafc")
        ax.set_facecolor("#ffffff")
        
        sns.boxplot(
            data=dataframe, 
            x=group_col, 
            y=numeric_col, 
            palette="viridis", 
            ax=ax,
            width=0.4,
            linewidth=1.5,
            fliersize=3
        )
        
        ax.set_title(f"{numeric_col} grouped by {group_col}", fontsize=11, fontweight="bold", pad=12, color="#0f172a")
        ax.set_xlabel(group_col, fontsize=8.5, color="#475569", labelpad=8)
        ax.set_ylabel(numeric_col, fontsize=8.5, color="#475569", labelpad=8)
        ax.tick_params(labelsize=8, colors="#64748b")
        plt.xticks(rotation=20, ha="right")
        
        ax.grid(True, linestyle="--", alpha=0.5, color="#e2e8f0")
        for spine in ["top", "right", "left", "bottom"]:
            ax.spines[spine].set_visible(False)
            
        plt.tight_layout()
        
        # Save chart copy to reports directory for report generator
        try:
            chart_path = Path(BASE_DIR) / "reports" / "active_chart.png"
            chart_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(chart_path, dpi=150, facecolor=fig.get_facecolor(), edgecolor="none", bbox_inches="tight")
        except Exception as save_err:
            logger.error(f"Failed to save active boxplot chart: {save_err}")
            
        if show_on_streamlit:
            st.pyplot(fig)
        plt.close(fig)
    except Exception as e:
        logger.error(f"Error rendering boxplot: {e}", exc_info=True)
        if show_on_streamlit:
            st.error("Failed to generate boxplot comparison visualization.")


def render_scatter_relationship(dataframe: pd.DataFrame, x_col: str, y_col: str, hue_col: str = None, show_on_streamlit: bool = True) -> None:
    """Renders a Seaborn Scatter Plot checking correlation of two parameters."""
    try:
        if dataframe.empty or x_col not in dataframe.columns or y_col not in dataframe.columns:
            if show_on_streamlit:
                st.info("No correlation columns available.")
            return

        fig, ax = plt.subplots(figsize=(7, 3.5), facecolor="#f8fafc")
        ax.set_facecolor("#ffffff")
        
        sns.scatterplot(
            data=dataframe, 
            x=x_col, 
            y=y_col, 
            hue=hue_col, 
            palette="crest" if hue_col else None,
            alpha=0.8, 
            ax=ax,
            edgecolor="none",
            s=40
        )
        
        ax.set_title(f"Correlation: {x_col} vs {y_col}", fontsize=11, fontweight="bold", pad=12, color="#0f172a")
        ax.set_xlabel(x_col, fontsize=8.5, color="#475569", labelpad=8)
        ax.set_ylabel(y_col, fontsize=8.5, color="#475569", labelpad=8)
        ax.tick_params(labelsize=8, colors="#64748b")
        
        if hue_col and ax.get_legend():
            plt.legend(title=hue_col, title_fontsize=8, fontsize=7.5, loc="best", framealpha=0.8, edgecolor="#e2e8f0")
            
        ax.grid(True, linestyle="--", alpha=0.5, color="#e2e8f0")
        for spine in ["top", "right", "left", "bottom"]:
            ax.spines[spine].set_visible(False)
            
        plt.tight_layout()
        if show_on_streamlit:
            st.pyplot(fig)
        plt.close(fig)
    except Exception as e:
        logger.error(f"Error rendering scatter plot: {e}", exc_info=True)
        if show_on_streamlit:
            st.error("Failed to generate scatter correlation plot.")




# ==============================================================================
# PIPELINE INGESTION CACHING
# ==============================================================================

@st.cache_data(show_spinner="Safely ingesting CSV file...")
def get_cached_dataframe(file_path: Path, file_hash: str) -> pd.DataFrame:
    """Cache loaded DataFrame to minimize redundant disk IO, invalidated on file hash change."""
    return load_csv(file_path)


@st.cache_data(show_spinner="Extracting structural schema boundaries...")
def get_cached_metadata(dataframe: pd.DataFrame, file_hash: str) -> Dict[str, Any]:
    """Cache extracted metadata structures, invalidated on file hash change."""
    return extract_schema_metadata(dataframe)


@st.cache_data(show_spinner="Querying AI for visualization layout...")
def get_cached_recommendations(schema_metadata: Dict[str, Any], file_hash: str, api_key: Optional[str]) -> Dict[str, Any]:
    """Cache LLM visualization recommendations, invalidated on file hash change."""
    from data_engine import recommend_visualizations
    return recommend_visualizations(schema_metadata, api_key)


# ==============================================================================
# MAIN APP EXECUTION
# ==============================================================================

def main() -> None:
    logger.info("Application starting up in Streamlit environment.")
    
    # Render layout Headers
    st.markdown('<h1 class="main-header">NGO Data Analytics Platform</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Secure, privacy-compliant analysis dashboard with hybrid conversational queries.</p>', unsafe_allow_html=True)
    
    # 1. Sidebar File Uploader for Custom Sheets
    st.sidebar.markdown("<h2 style='font-family: Outfit; font-weight: 600; color: #1e293b;'>Dataset Ingestion</h2>", unsafe_allow_html=True)
    uploaded_file = st.sidebar.file_uploader(
        "Upload custom CSV sheet", 
        type=["csv"], 
        help="Upload a new CSV dataset to analyze. If empty, the default dataset is used."
    )
    
    dataset_file = None
    file_hash = ""
    
    if uploaded_file is not None:
        file_bytes = uploaded_file.getvalue()
        file_hash = hashlib.md5(file_bytes).hexdigest()
        dataset_file = Path(BASE_DIR) / "data" / "uploaded_dataset.csv"
        
        # Save file locally if it's new or content changed
        try:
            if not dataset_file.exists() or hashlib.md5(dataset_file.read_bytes()).hexdigest() != file_hash:
                dataset_file.parent.mkdir(parents=True, exist_ok=True)
                dataset_file.write_bytes(file_bytes)
                logger.info(f"Saved uploaded dataset to {dataset_file} (MD5: {file_hash})")
        except Exception as save_err:
            st.error(f"Failed to ingest custom CSV file: {save_err}")
            return
    else:
        # Fall back to default path
        dataset_file = Path(settings.DATASET_PATH)
        if dataset_file.exists():
            try:
                file_hash = hashlib.md5(dataset_file.read_bytes()).hexdigest()
            except Exception:
                file_hash = "default"
        else:
            # Graceful landing instructions, waiting for user file upload
            st.markdown("""
            <div class="welcome-card">
                <h2 style='font-family: Outfit; font-weight: 800; color: #0f172a; margin-top: 0;'>👋 Welcome to the NGO Data Analytics Hub</h2>
                <p style='color: #475569; font-size: 1.1rem; line-height: 1.6;'>
                    This platform provides secure, privacy-preserving, and AI-driven visual analytics on participant health cohorts.
                </p>
                <div style='margin-top: 2rem; margin-bottom: 2rem;'>
                    <div class="step-container">
                        <div class="step-badge">1</div>
                        <div class="step-text">
                            <strong>Ingest your dataset</strong>: Upload a CSV dataset sheet using the <strong>Dataset Ingestion</strong> file uploader in the sidebar on the left.
                        </div>
                    </div>
                    <div class="step-container">
                        <div class="step-badge">2</div>
                        <div class="step-text">
                            <strong>Analyze</strong>: The engine will dynamically parse column metadata, identify categories, and initialize visual dashboard panels.
                        </div>
                    </div>
                    <div class="step-container">
                        <div class="step-badge">3</div>
                        <div class="step-text">
                            <strong>Query & Export</strong>: Ask conversational questions, filter metrics, view AI-recommended charts, and export executive PDF summaries.
                        </div>
                    </div>
                </div>
                <p style='color: #64748b; font-size: 0.9rem; font-style: italic; border-top: 1px solid #e2e8f0; padding-top: 1.25rem;'>
                    Note: To leverage conversational queries or LLM report summaries, configure your OpenRouter API Key in the sidebar override or in your environment settings.
                </p>
            </div>
            """, unsafe_allow_html=True)
            return
            
    # Reset filters if dataset changed to avoid slider bounds ValueError crash
    if "current_file_hash" not in st.session_state or st.session_state.current_file_hash != file_hash:
        logger.info("New dataset file hash detected. Clearing session state filters.")
        st.session_state.current_file_hash = file_hash
        # Clear all state variables except credentials override to force clean re-initialization
        for key in list(st.session_state.keys()):
            if key not in ("current_file_hash", "api_key_override"):
                del st.session_state[key]
        st.rerun()

    try:
        # Load and validate dataset using cached functions tracking the hash
        df_original = get_cached_dataframe(dataset_file, file_hash)
        if not validate_dataset(df_original):
            st.error("❌ The loaded dataset structure failed validation checks. Please review application logs.")
            return
            
        # Extract metadata
        metadata = get_cached_metadata(df_original, file_hash)
        
        # Initialize state variables
        initialize_session_state(metadata)
        
        # Classify columns for widgets, dynamic KPIs, and overrides
        numeric_cols = [col for col, info in metadata.items() if "number" in info["dtype"] or "int" in info["dtype"] or "float" in info["dtype"]]
        group_cols = [col for col, info in metadata.items() if "object" in info["dtype"] or "str" in info["dtype"] or "string" in info["dtype"] or "category" in info["dtype"] or "bool" in info["dtype"]]
        
        # ==============================================================================
        # SIDEBAR PANEL - DETERMINISTIC CONTROLS & DIAGNOSTICS
        # ==============================================================================
        st.sidebar.markdown("<h2 style='font-family: Outfit; font-weight: 600; color: #1e293b;'>Control Panel</h2>", unsafe_allow_html=True)
        st.sidebar.markdown("---")
        
        # 1. API key override
        st.sidebar.subheader("API Configuration")
        sidebar_key = st.sidebar.text_input(
            "OpenRouter API Key Override", 
            type="password", 
            value=st.session_state.api_key_override,
            help="Specify custom API key(s). Supports a comma-separated list for key rotation (e.g. key1,key2)."
        )
        if sidebar_key:
            settings.OPENROUTER_API_KEY = sidebar_key
            st.session_state.api_key_override = sidebar_key
            
        api_key_active = check_any_key_configured(st.session_state.api_key_override)
        if api_key_active:
            # Run cached connectivity handshake
            connection_ok = check_api_connection(st.session_state.api_key_override)
            if connection_ok:
                st.sidebar.markdown('<span class="badge badge-success">🟢 AI Active (Handshake OK)</span>', unsafe_allow_html=True)
            else:
                st.sidebar.markdown('<span class="badge badge-error">🔴 AI Handshake Failed</span>', unsafe_allow_html=True)
        else:
            st.sidebar.markdown('<span class="badge badge-warning">🟡 AI Offline (Key Missing)</span>', unsafe_allow_html=True)
            
        st.sidebar.markdown("---")
        
        # 2. Hybrid Filters widget
        st.sidebar.subheader("Deterministic Filters")
        
        # Initialize widget variables to safe defaults to avoid NameErrors
        selected_genders = []
        selected_diets = []
        selected_workouts = []
        selected_age_range = (18, 60)
        selected_weight_range = (40, 120)
        age_min_limit, age_max_limit = 18, 60
        weight_min_limit, weight_max_limit = 40, 120
        
        has_any_filters = False
        
        # Age slider
        if "Age" in metadata:
            has_any_filters = True
            age_min_limit = int(metadata.get("Age", {}).get("range", {}).get("min", 18))
            age_max_limit = int(metadata.get("Age", {}).get("range", {}).get("max", 60))
            selected_age_range = st.sidebar.slider(
                "Age Range",
                min_value=age_min_limit,
                max_value=age_max_limit,
                value=(age_min_limit, age_max_limit),
                key="widget_age_range"
            )
            
        # Weight slider
        if "Weight (kg)" in metadata:
            has_any_filters = True
            weight_min_limit = int(metadata.get("Weight (kg)", {}).get("range", {}).get("min", 40))
            weight_max_limit = int(metadata.get("Weight (kg)", {}).get("range", {}).get("max", 120))
            selected_weight_range = st.sidebar.slider(
                "Weight Range (kg)",
                min_value=weight_min_limit,
                max_value=weight_max_limit,
                value=(weight_min_limit, weight_max_limit),
                key="widget_weight_range"
            )
            
        # Gender selection
        if "Gender" in metadata:
            has_any_filters = True
            gender_options = metadata.get("Gender", {}).get("unique_values", [])
            selected_genders = st.sidebar.multiselect(
                "Select Genders", 
                options=gender_options, 
                default=[],
                key="widget_genders"
            )
            
        # Diet Selection
        if "diet_type" in metadata:
            has_any_filters = True
            diet_options = metadata.get("diet_type", {}).get("unique_values", [])
            selected_diets = st.sidebar.multiselect(
                "Select Diets", 
                options=diet_options, 
                default=[],
                key="widget_diets"
            )
            
        # Workout type selection
        if "Workout_Type" in metadata:
            has_any_filters = True
            workout_options = metadata.get("Workout_Type", {}).get("unique_values", [])
            selected_workouts = st.sidebar.multiselect(
                "Select Workout Types", 
                options=workout_options, 
                default=[],
                key="widget_workouts"
            )
            
        # Reset filters button
        if has_any_filters:
            if st.sidebar.button("Clear All Sidebar Filters", use_container_width=True):
                logger.info("Resetting all sidebar widgets to defaults.")
                if "widget_age_range" in st.session_state:
                    st.session_state.widget_age_range = (age_min_limit, age_max_limit)
                if "widget_weight_range" in st.session_state:
                    st.session_state.widget_weight_range = (weight_min_limit, weight_max_limit)
                if "widget_genders" in st.session_state:
                    st.session_state.widget_genders = []
                if "widget_diets" in st.session_state:
                    st.session_state.widget_diets = []
                if "widget_workouts" in st.session_state:
                    st.session_state.widget_workouts = []
                st.rerun()
        else:
            st.sidebar.info("No standard filters available for this dataset.")
            
        st.sidebar.markdown("---")
        
        # 3. Environment diagnostics
        st.sidebar.subheader("System Status")
        st.sidebar.markdown(f"**Mode**: `{settings.ENV.upper()}`")
        st.sidebar.markdown(f"**Dataset Rows**: `{len(df_original):,}`")
        st.sidebar.markdown(f"**Log Path**: `logs/app.log`")
        
        # ==============================================================================
        # CONVERSATIONAL QUERY CANVAS
        # ==============================================================================
        st.markdown("### 💬 Conversational Query")
        st.markdown(
            "*Type an analysis query in natural language. The system translates request to local filters "
            "without uploading dataset rows.*"
        )
        
        col_nl, col_act = st.columns([4, 1])
        
        with col_nl:
            query_val = st.text_input(
                "Search Query", 
                value=st.session_state.nl_query, 
                placeholder="e.g., Show me fitness metrics for vegan males under 30",
                label_visibility="collapsed",
                disabled=not api_key_active
            )
            
        with col_act:
            btn_col1, btn_col2 = st.columns(2)
            with btn_col1:
                submit_query = st.button("Submit", type="primary", use_container_width=True, disabled=not api_key_active)
            with btn_col2:
                clear_query = st.button("Clear", use_container_width=True)
                
        # Suggestion Prompts
        st.markdown('<p style="font-size:0.85rem; color:#64748b; margin-top:-0.5rem;">💡 <b>Try:</b> '
                    '<span style="cursor:pointer; color:#2563eb; text-decoration:underline;" id="sugg1">"vegan males under 30"</span> | '
                    '<span style="cursor:pointer; color:#2563eb; text-decoration:underline;" id="sugg2">"individuals with weight > 80kg and low workout frequency"</span></p>', 
                    unsafe_allow_html=True)
        
        # Handle query clear trigger
        if clear_query:
            logger.info("Clearing active Conversational Filters.")
            st.session_state.nl_query = ""
            st.session_state.nl_payload = {"filters": []}
            st.rerun()
            
        # Handle query submission
        if submit_query and query_val:
            st.session_state.nl_query = query_val
            with st.spinner("Translating natural language into validated filters..."):
                try:
                    payload = parse_natural_language_query(
                        query_val, 
                        metadata, 
                        api_key=st.session_state.api_key_override
                    )
                    if "error" in payload:
                        st.error(f"❌ LLM translation failed: {payload['error']}")
                    else:
                        st.session_state.nl_payload = payload
                except Exception as e:
                    logger.exception("Conversational parsing failed:")
                    st.error("An unexpected error occurred during NLP query processing.")
                    
        # ==============================================================================
        # HYBRID FILTERING MERGE LAYER
        # ==============================================================================
        # Read widget variables dynamically
        sidebar_filters = []
        
        # Categorical lists with active metadata check
        if "Gender" in metadata and selected_genders:
            sidebar_filters.append({"column": "Gender", "operator": "in", "value": selected_genders})
        if "diet_type" in metadata and selected_diets:
            sidebar_filters.append({"column": "diet_type", "operator": "in", "value": selected_diets})
        if "Workout_Type" in metadata and selected_workouts:
            sidebar_filters.append({"column": "Workout_Type", "operator": "in", "value": selected_workouts})
            
        # Numeric sliders (only append if different from baseline metadata min/max)
        if "Age" in metadata:
            if selected_age_range[0] > age_min_limit:
                sidebar_filters.append({"column": "Age", "operator": ">=", "value": selected_age_range[0]})
            if selected_age_range[1] < age_max_limit:
                sidebar_filters.append({"column": "Age", "operator": "<=", "value": selected_age_range[1]})
            
        if "Weight (kg)" in metadata:
            if selected_weight_range[0] > weight_min_limit:
                sidebar_filters.append({"column": "Weight (kg)", "operator": ">=", "value": selected_weight_range[0]})
            if selected_weight_range[1] < weight_max_limit:
                sidebar_filters.append({"column": "Weight (kg)", "operator": "<=", "value": selected_weight_range[1]})
            
        # Merge both filter sources
        nl_filters = st.session_state.nl_payload.get("filters", [])
        combined_filters = nl_filters + sidebar_filters
        combined_payload = {"filters": combined_filters}
        
        # Display active filters summary if present
        if combined_filters:
            with st.expander("🛠️ Active Ingestion Filters Details", expanded=False):
                st.write("**Parsed JSON Structure:**")
                st.json(combined_payload)
                
        # Validate and Execute securely
        try:
            with st.spinner("Processing filters locally..."):
                if validate_filter_payload(combined_payload, df_original):
                    filtered_df = apply_filters(df_original, combined_payload)
                else:
                    st.error("❌ **Security Alert**: The combined filters failed local safety checks and cannot be executed.")
                    filtered_df = pd.DataFrame()
        except Exception as filter_err:
            logger.exception("Safe execution pipeline error:")
            st.error("A validation exception occurred. Reverting to empty subset.")
            filtered_df = pd.DataFrame()
            
        # ==============================================================================
        # CORE DASHBOARD TABS
        # ==============================================================================
        tab_analytics, tab_preview, tab_definitions, tab_reports = st.tabs([
            "📊 Active Analytics",
            "📋 Data Preview",
            "🔍 Metadata Schema",
            "📄 Export Reports"
        ])
        
        # Compute aggregate statistics
        analytics = generate_analytics(filtered_df)
        matched_count = len(filtered_df)
        
        # --- TAB 1: ACTIVE ANALYTICS ---
        with tab_analytics:
            # 1. Render Premium Dynamic KPI Cards
            # Filter out ID-like columns to avoid displaying meaningless stats
            displayable_cols = [
                col for col in numeric_cols 
                if col.lower() not in ["beneficiary_id", "id", "ssn", "index", "unnamed: 0"]
            ]
            
            # Determine how many columns we can display (max 6, including Matched Samples)
            num_cards = min(len(displayable_cols) + 1, 6)
            kpi_cols = st.columns(num_cards)
            
            # Matched rows card (always first)
            with kpi_cols[0]:
                st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-label">Matched Samples</div>
                    <div class="kpi-value">{matched_count:,}</div>
                </div>
                """, unsafe_allow_html=True)
                
            # Render numerical aggregates dynamically
            kpi_index = 1
            for col in displayable_cols:
                if kpi_index >= 6:
                    break
                    
                avg_val = "N/A"
                if matched_count > 0:
                    val = filtered_df[col].mean()
                    # Format unit suffix if name implies it
                    unit = ""
                    col_lower = col.lower()
                    if "protein" in col_lower or "carb" in col_lower or "fat" in col_lower:
                        unit = "g"
                    elif "kg" in col_lower or "weight" in col_lower:
                        unit = " kg"
                    elif "hour" in col_lower or "duration" in col_lower:
                        unit = "h"
                    
                    if pd.isna(val):
                        avg_val = "N/A"
                    else:
                        try:
                            f_val = float(val)
                            import numpy as np
                            if np.isnan(f_val) or np.isinf(f_val):
                                avg_val = "N/A"
                            elif f_val.is_integer():
                                avg_val = f"{int(f_val):,}{unit}"
                            else:
                                avg_val = f"{f_val:,.1f}{unit}"
                        except (ValueError, TypeError):
                            avg_val = "N/A"
                
                # Format label dynamically
                display_label = f"Avg {col}"
                if any(x in col_lower for x in ["avg", "mean", "average"]):
                    display_label = col
                    
                with kpi_cols[kpi_index]:
                    st.markdown(f"""
                    <div class="kpi-card">
                        <div class="kpi-label">{display_label}</div>
                        <div class="kpi-value">{avg_val}</div>
                    </div>
                    """, unsafe_allow_html=True)
                kpi_index += 1
                
            st.markdown("<br>", unsafe_allow_html=True)
            
            if matched_count == 0:
                st.warning("⚠️ **Empty Filter Result**: No records match the combined filter criteria. Try expanding range widgets or clearing conversational queries.")
            else:
                # 2. Get Dynamic Graph Recommendations (100% local and deterministic)
                recommendations_payload = get_deterministic_graph_recommendations(metadata)
                recommendations = recommendations_payload.get("recommendations", [])
                
                # Render Recommended Dashboard
                st.markdown("#### 📈 Dynamic Visual Analytics")
                st.markdown(
                    "*The system analyzed the dataset's schema and identified the following three optimized "
                    "visualizations to uncover program trends.*"
                )
                
                col_left, col_right = st.columns(2)
                
                with col_left:
                    # Chart 1: Distribution
                    if len(recommendations) > 0:
                        rec = recommendations[0]
                        st.markdown(f"##### **1. {rec.get('title')}**")
                        if rec.get('description'):
                            st.markdown(f"<p style='font-size:0.85rem; color:#475569; margin-top:-0.5rem;'>💡 <i>{rec.get('description')}</i></p>", unsafe_allow_html=True)
                        x_col = rec.get("x_column")
                        if x_col in filtered_df.columns:
                            render_distribution_plot(filtered_df, x_col)
                        else:
                            st.info(f"Recommended column '{x_col}' is missing.")
                            
                    st.markdown("<br>", unsafe_allow_html=True)
                    
                    # Chart 2: Grouped Box Plot
                    if len(recommendations) > 1:
                        rec = recommendations[1]
                        st.markdown(f"##### **2. {rec.get('title')}**")
                        if rec.get('description'):
                            st.markdown(f"<p style='font-size:0.85rem; color:#475569; margin-top:-0.5rem;'>💡 <i>{rec.get('description')}</i></p>", unsafe_allow_html=True)
                        x_col = rec.get("x_column")
                        y_col = rec.get("y_column")
                        if x_col in filtered_df.columns and y_col in filtered_df.columns:
                            render_group_boxplot(filtered_df, y_col, x_col)
                        else:
                            st.info(f"Recommended comparison columns ('{x_col}', '{y_col}') are missing.")
                            
                with col_right:
                    # Chart 3: Correlation Scatter Plot
                    if len(recommendations) > 2:
                        rec = recommendations[2]
                        st.markdown(f"##### **3. {rec.get('title')}**")
                        if rec.get('description'):
                            st.markdown(f"<p style='font-size:0.85rem; color:#475569; margin-top:-0.5rem;'>💡 <i>{rec.get('description')}</i></p>", unsafe_allow_html=True)
                        x_col = rec.get("x_column")
                        y_col = rec.get("y_column")
                        hue_col = rec.get("hue_column")
                        if x_col in filtered_df.columns and y_col in filtered_df.columns:
                            hue_var = hue_col if (hue_col and hue_col in filtered_df.columns and hue_col != "None") else None
                            render_scatter_relationship(filtered_df, x_col, y_col, hue_var)
                        else:
                            st.info(f"Recommended correlation columns ('{x_col}', '{y_col}') are missing.")
                            
                # Custom Manual Overrides Expander
                st.markdown("<br>", unsafe_allow_html=True)
                with st.expander("🔧 Custom Visualization Overrides (Manual Selectors)", expanded=False):
                    st.markdown("Configure manual overrides to customize the variables displayed on the dashboard:")
                    
                    v_col1, v_col2 = st.columns(2)
                    chart_col = None
                    if numeric_cols:
                        with v_col1:
                            chart_col = st.selectbox(
                                "Select Custom Distribution Variable (KDE + Hist):", 
                                options=numeric_cols,
                                index=numeric_cols.index("Age") if "Age" in numeric_cols else 0,
                                key="selected_chart_col"
                            )
                            render_distribution_plot(filtered_df, chart_col)
                    else:
                        st.info("No numeric columns available in the dataset for distribution plotting.")
                        
                    if numeric_cols and group_cols and chart_col:
                        with v_col2:
                            groupby_col = st.selectbox(
                                "Select Custom Grouping Category (Box Plot):",
                                options=group_cols,
                                index=group_cols.index("Gender") if "Gender" in group_cols else 0,
                                key="selected_groupby_col"
                            )
                            render_group_boxplot(filtered_df, chart_col, groupby_col)
                    else:
                        st.info("No grouping category columns available in the dataset for boxplot comparison.")
                        
                    st.markdown("---")
                    st.markdown("##### Custom Correlation Scatter Plot")
                    if len(numeric_cols) >= 2:
                        corr_col1, corr_col2, corr_col3 = st.columns(3)
                        
                        with corr_col1:
                            x_axis_var = st.selectbox(
                                "X-Axis Variable", 
                                options=numeric_cols,
                                index=numeric_cols.index("Session_Duration (hours)") if "Session_Duration (hours)" in numeric_cols else 0
                            )
                        with corr_col2:
                            y_axis_var = st.selectbox(
                                "Y-Axis Variable", 
                                options=numeric_cols,
                                index=numeric_cols.index("Calories_Burned") if "Calories_Burned" in numeric_cols else 0
                            )
                        with corr_col3:
                            hue_axis_var = st.selectbox(
                                "Color / Legend Variable", 
                                options=["None"] + group_cols,
                                index=1 if "Gender" in group_cols else 0
                            )
                            
                        hue_var = None if hue_axis_var == "None" else hue_axis_var
                        render_scatter_relationship(filtered_df, x_axis_var, y_axis_var, hue_var)
                    else:
                        st.info("At least two numeric columns are required in the dataset to plot correlations.")
                
        # --- TAB 2: DATA PREVIEW ---
        with tab_preview:
            st.markdown("### 📋 Filtered Subsplit Preview")
            st.markdown("Displaying the top 10 rows matching your current filters. PII indicators are stripped to protect privacy.")
            
            if matched_count == 0:
                st.info("No matching records to preview.")
            else:
                # Private Scrubbing: Exclude IDs and personal trackers if any exist
                sensitive_columns = ["Beneficiary_ID", "Name", "ID", "SSN", "Name of Exercise"]
                preview_df = filtered_df.head(10).copy()
                
                for sensitive_col in sensitive_columns:
                    if sensitive_col in preview_df.columns:
                        preview_df = preview_df.drop(columns=[sensitive_col])
                        
                st.dataframe(preview_df, use_container_width=True)
                
        # --- TAB 3: METADATA DEFINITIONS ---
        with tab_definitions:
            st.markdown("### 🔍 Active Schema Definitions")
            st.markdown("Columns metadata, pandas types, and completeness statistics:")
            
            # Map structural components
            col_list = []
            for col_name, info in get_cached_metadata(df_original, file_hash).items():
                col_list.append({
                    "Variable name": col_name,
                    "Dtype": info["dtype"],
                    "Null Count": info["null_count"],
                    "Completeness Percentage": f"{100.0 - info['null_percentage']:.2f}%"
                })
            st.dataframe(pd.DataFrame(col_list), use_container_width=True, hide_index=True)
            
        # --- TAB 4: EXPORT REPORTS ---
        with tab_reports:
            st.markdown("### 📄 Export PDF Executive Reports")
            st.markdown(
                "Generate a formal, publication-ready one-page Executive Summary containing "
                "active filter schemas, target health KPIs, narrative AI recommendations, "
                "and visual correlation charts."
            )
            
            if matched_count == 0:
                st.warning("⚠️ Cannot generate report: Match set is empty. Modify filters in the sidebar first.")
            else:
                # API Warnings
                if not api_key_active:
                    st.info("ℹ️ OpenRouter Key is missing. The executive summary report will compile using deterministic, locally calculated fallback insights.")
                    
                if st.button("Generate Executive Report", type="primary", use_container_width=True):
                    # Progress triggers
                    # 1. Fetch AI Graph Recommendations (only triggered during report generation)
                    with st.spinner("1. Querying OpenRouter to select optimized report visual chart..."):
                        recommendations_payload = get_cached_recommendations(
                            metadata, 
                            file_hash, 
                            api_key=st.session_state.api_key_override
                        )
                        recommendations = recommendations_payload.get("recommendations", [])
                        
                    # 2. Render recommended chart in background
                    with st.spinner("2. Rendering report visual chart in background..."):
                        rendered_background = False
                        if recommendations:
                            rec = recommendations[0]  # Take primary recommendation
                            rec_type = rec.get("type")
                            x_col = rec.get("x_column")
                            y_col = rec.get("y_column")
                            hue_col = rec.get("hue_column")
                            
                            # Render safely based on type
                            if rec_type == "distribution" and x_col in filtered_df.columns:
                                render_distribution_plot(filtered_df, x_col, show_on_streamlit=False)
                                rendered_background = True
                            elif rec_type == "boxplot" and x_col in filtered_df.columns and y_col in filtered_df.columns:
                                render_group_boxplot(filtered_df, y_col, x_col, show_on_streamlit=False)
                                rendered_background = True
                            elif rec_type == "correlation" and x_col in filtered_df.columns and y_col in filtered_df.columns:
                                hue_var = hue_col if (hue_col and hue_col in filtered_df.columns and hue_col != "None") else None
                                render_scatter_relationship(filtered_df, x_col, y_col, hue_var, show_on_streamlit=False)
                                rendered_background = True
                                
                        if not rendered_background:
                            # Fallback to dynamic distribution chart in background
                            displayable_cols = [
                                col for col in numeric_cols 
                                if col.lower() not in ["beneficiary_id", "id", "ssn", "index", "unnamed: 0"]
                            ]
                            if displayable_cols:
                                render_distribution_plot(filtered_df, displayable_cols[0], show_on_streamlit=False)

                    with st.spinner("3. Querying OpenRouter Public Health Analyst for insights..."):
                        insights = generate_executive_insights(
                            analytics, 
                            api_key=st.session_state.api_key_override
                        )
                        
                    with st.spinner("4. Compiling ReportLab PDF layout page..."):
                        # Define path to active chart saved during background Matplotlib execution
                        chart_file_path = str(Path(BASE_DIR) / "reports" / "active_chart.png")
                        try:
                            pdf_data = generate_executive_report_pdf(
                                filters=combined_payload,
                                analytics_summary=analytics,
                                insights=insights,
                                chart_path=chart_file_path
                            )
                            
                            # Run end-to-end integration validation
                            check_results = verify_reporting_pipeline()
                            
                            if check_results["download_ready"]:
                                st.success("🎉 Executive Report compiled successfully!")
                                
                                # Render insight list preview
                                st.markdown("#### Narrative Insights Preview")
                                for idx, ins in enumerate(insights):
                                    st.markdown(f"**{idx+1}.** {ins}")
                                    
                                # download button
                                st.download_button(
                                    label="💾 Save PDF Executive Report",
                                    data=pdf_data,
                                    file_name=f"NGO_Executive_Report_{time.strftime('%Y-%m-%d')}.pdf",
                                    mime="application/pdf",
                                    use_container_width=True
                                )
                            else:
                                st.error("❌ The generated PDF document failed structural verification tests.")
                        except Exception as pdf_compile_err:
                            logger.exception("PDF report compiler failed:")
                            st.error(f"❌ PDF generation failed: {pdf_compile_err}")
                            
    except Exception as general_err:
        logger.exception("General application exception:")
        st.error("❌ **Critical Application Error**: Streamlit encountered an unexpected exception. Please check the logs.")


if __name__ == "__main__":
    main()
