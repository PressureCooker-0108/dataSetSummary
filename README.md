# NGO Data Analytics Hub

A production-grade, secure, and observable data analytics platform foundation for NGOs. This workspace ingests structured CSV datasets, validates completeness and schemas, generates metadata summary reports, and serves a Streamlit-based monitoring dashboard, following strict data privacy compliance.

It includes a secure **Natural Language Query Engine** powered by `gpt-4o-mini` that translates conversational user questions into local pandas filters without exposing raw records to external APIs or executing dynamic code.

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Conversational Query Engine](#conversational-query-engine)
3. [Token Cost Optimization](#token-cost-optimization)
4. [Project Structure](#project-structure)
5. [Environment Setup & Installation](#environment-setup--installation)
6. [Data Ingestion & Inherent Privacy](#data-ingestion--inherent-privacy)
7. [Logging Architecture](#logging-architecture)
8. [Streamlit Execution](#streamlit-execution)
9. [Docker Deployment](#docker-deployment)
10. [Testing & Verification](#testing--verification)
11. [Troubleshooting Guide](#troubleshooting-guide)

---

## Project Overview

NGOs often handle sensitive beneficiary, wellness, or socio-economic datasets. This workspace provides a robust template designed to safely load datasets, inspect columns, evaluate missing (null) values, calculate duplicates, and print metadata reports **without exposing raw row-level records**.

---

## Conversational Query Engine

The natural language query engine translates user queries into pandas filters locally.

### Security Boundary Model
1. **Zero Raw Rows Transmitted**: External APIs (like OpenAI) only see high-level metadata (column names, datatypes, numerical ranges, and unique categories for low-cardinality columns).
2. **No Dynamic Execution (`eval()` / `exec()`)**: Prohibits code string injection. The engine maps filters to a JSON schema, parses them, and builds pandas boolean masks programmatically:
   ```python
   mask &= (dataframe[col] < val)
   ```
3. **Strict Validation Layer**:
   - Matches columns against the dataframe columns.
   - Restricts operators to a whitelisted set: `["==", "!=", ">", ">=", "<", "<=", "in", "not in"]`.
   - Casts value types to match pandas column datatypes, rejecting malformed injection attempts.

---

## Token Cost Optimization

To minimize operating costs and latency when interacting with OpenAI, the engine implements several optimizations:

- **Metadata-Only Payloads**: Only column data schemas are sent to the LLM. 20,000+ data rows remain local and are never parsed by the model.
- **Low-Cardinality Categorical Previews**: Rather than list all unique text categories, lists of distinct values are only extracted and attached for variables with $\le 15$ unique values (e.g. `Gender`, `Diet`). Larger cardinality columns (like `meal_name`) only expose their datatype and null count.
- **Numeric Range Compaction**: Continuous variables (like `Age`, `Calories`) only list their float ranges (`min` and `max`) rather than value distributions.
- **Simplified Data Dictionaries**: High-level metadata dictionary structures are minimized before API calls to drop unused keys (null counts, percentage floats) that do not benefit the translation task.

---

## Project Structure

```
dataSetSummarizer/
├── app.py                     # Streamlit frontend entrypoint (with NL Query tab)
├── data_engine.py             # Safe ingestion, validation, and NL Query layer
├── requirements.txt           # Python package dependencies
├── README.md                  # Documentation and setup instructions
├── .gitignore                 # Files excluded from version control
├── .env.example               # Environment variables template
├── .env                       # Local environment configurations (private)
├── Dockerfile                 # Container image specification
├── .dockerignore              # Docker build context filters
├── config/
│   ├── __init__.py
│   └── settings.py            # Settings validation via Pydantic
├── data/
│   └── ONAM EXCEL.csv         # Source CSV dataset (ignored in git)
├── logs/
│   └── app.log                # Application rotating log file (ignored in git)
├── reports/
│   └── summary_report.pdf     # Generated PDF metadata reports (ignored in git)
├── tests/
│   ├── __init__.py
│   └── test_data_engine.py    # Pytest unit tests suite
└── utils/
    ├── __init__.py
    └── logger.py              # Central rotating log setup
```

---

## Environment Setup & Installation

### Windows Powershell Setup (Local Developer)

1. **Clone/Open Workspace**:
   Navigate to your project folder:
   ```powershell
   cd c:\Users\Adity\OneDrive\Desktop\Projects\dataSetSummarizer
   ```

2. **Create Python Virtual Environment**:
   ```powershell
   python -m venv .venv
   ```

3. **Activate Environment**:
   ```powershell
   .venv\Scripts\Activate.ps1
   ```

4. **Install Dependencies**:
   ```powershell
   pip install -r requirements.txt
   ```

5. **Configure Environments**:
   Copy the example environment settings to the active `.env` file:
   ```powershell
   Copy-Item -Path ".env.example" -Destination ".env"
   ```
   Open the `.env` file and customize your `OPENAI_API_KEY`.

---

## Data Ingestion & Inherent Privacy

To protect beneficiary records, the application adheres to strict data privacy:
- **Zero Raw Exposures**: Raw table rows are never printed to the logs or console, nor are they displayed in raw tabular format in Streamlit.
- **Robust Loading**: `load_csv()` automatically attempts encoding fallback (`utf-8`, `latin-1`, `cp1252`, `utf-16`) to guarantee successful ingestions of legacy files.
- **Auto-Validation**: The data engine verifies shape dimension constraints, log load performance metrics, and detects duplicate rows automatically.

---

## Logging Architecture

Observable errors, performance logs, and events are processed centrally in `utils/logger.py` and output to both standard output (console) and a rotating file (`logs/app.log`).

- **Rotations**: Files rotate at 5 MB each and retain up to 5 backups to prevent disk depletion.
- **Structure**: Logs follow a timestamped format: `[Timestamp] [Level] [Module:File:Line] - Message`.
- **Capture Areas**:
  - Application start and shutdown hooks.
  - CSV loading latency metrics.
  - Null percentage distributions.
  - Conversational query filters execution telemetry.
  - Execution errors and warnings.

---

## Streamlit Execution

Run the Streamlit server from your active virtual environment:
```powershell
streamlit run app.py
```
By default, the dashboard will open at [http://localhost:8501](http://localhost:8501). Enter your `OPENAI_API_KEY` override directly in the sidebar control panel to test conversational queries.

---

## Docker Deployment

The application is dockerized using a lightweight, secure base image (`python:3.11-slim`) and runs under a non-privileged system user (`appuser`).

### Build Container:
```bash
docker build -t ngo-analytics-hub .
```

### Run Container:
```bash
docker run -p 8501:8501 --env-file .env ngo-analytics-hub
```

---

## Testing & Verification

Unit tests are written in Pytest to test logger setup, CSV loading failures, schema structure analysis, and ReportLab PDF compilation:

```powershell
.venv\Scripts\pytest -v
```

---

## Troubleshooting Guide

- **AttributeError: 'Settings' object has no attribute 'BASE_DIR'**:
  - *Solution*: Ensure you import `BASE_DIR` as a module constant (`from config.settings import BASE_DIR`) rather than calling it on the settings instance.
- **Missing Dataset Warning**:
  - *Solution*: Place your target CSV inside `data/ONAM EXCEL.csv` or edit the path variable `DATASET_PATH` in your `.env` file to point to your data source.
- **Pytest: ModuleNotFoundError**:
  - *Solution*: Run pytest through the virtual environment path `.venv\Scripts\pytest -v` or ensure your virtual environment is fully activated.
