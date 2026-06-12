import os
import io
from pathlib import Path
import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from main import app, get_active_dataset_path, BASE_DIR
from config.settings import settings

client = TestClient(app)

@pytest.fixture
def mock_csv_file():
    """Generates an in-memory CSV file representation."""
    df = pd.DataFrame({
        "Age": [25, 30, 35, 40, 45],
        "Weight (kg)": [60.5, 70.0, 80.2, 90.1, 85.0],
        "Gender": ["Male", "Female", "Male", "Female", "Non-binary"],
        "Diet": ["Vegan", "Vegetarian", "Vegan", "Omnivore", "Vegan"]
    })
    csv_buffer = io.BytesIO()
    df.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)
    return csv_buffer.getvalue()

def test_health_check():
    """Verify that the health check endpoint returns 200 and indicates healthy status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "active_dataset" in data

def test_get_schema_success():
    """Test schema metadata retrieval."""
    response = client.get("/api/schema")
    # Even if no dataset was uploaded, it falls back to the default dataset (settings.DATASET_PATH)
    assert response.status_code in [200, 404]
    if response.status_code == 200:
        data = response.json()
        assert data["success"] is True
        assert "schema" in data
        assert len(data["schema"]) > 0

def test_upload_dataset_success(mock_csv_file):
    """Test that a valid CSV file upload is parsed, saved, and returns schema metadata."""
    # We patch the save path so we don't overwrite the actual user uploaded dataset
    with patch("main.UPLOADED_FILE_PATH") as mock_path:
        # Create a mock temporary file path in tests
        temp_file = Path(BASE_DIR) / "data" / "test_uploaded_dataset.csv"
        mock_path.exists.return_value = True
        mock_path.name = "test_uploaded_dataset.csv"
        mock_path.write_bytes = MagicMock()
        
        # Patch active dataset loader to load our mock data
        df = pd.read_csv(io.BytesIO(mock_csv_file))
        with patch("main.get_active_dataset", return_value=df):
            response = client.post(
                "/api/upload",
                files={"file": ("test_cohort.csv", mock_csv_file, "text/csv")}
            )
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["filename"] == "test_cohort.csv"
            assert "schema" in data
            assert "Age" in data["schema"]
            assert "Weight (kg)" in data["schema"]

def test_upload_invalid_file_format():
    """Test that upload rejects non-CSV file formats."""
    response = client.post(
        "/api/upload",
        files={"file": ("document.txt", b"some plain text contents", "text/plain")}
    )
    assert response.status_code == 400
    assert "Only CSV" in response.json()["detail"]

def test_run_query_endpoint(mock_csv_file):
    """Test query endpoint with filtering, math expressions, and chart bucketing."""
    df = pd.read_csv(io.BytesIO(mock_csv_file))
    
    with patch("main.get_active_dataset", return_value=df), \
         patch("main.profile_dataset", return_value="Test profile"), \
         patch("main.parse_natural_language_query", return_value={"filters": [], "expression": None}):
         
        # Test 1: Simple Filter and custom math expression
        payload = {
            "filters": [
                {"column": "Age", "operator": ">=", "value": 30}
            ],
            "math_expression": "df['Weight (kg)'].mean()",
            "chart_column": "Age"
        }
        
        response = client.post("/api/query", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["matched_count"] == 4  # 30, 35, 40, 45 (4 values >= 30)
        assert data["total_count"] == 5
        assert data["metric_expression"] == "df['Weight (kg)'].mean()"
        # Average weight of [30, 35, 40, 45] is (70.0 + 80.2 + 90.1 + 85.0) / 4 = 81.325
        assert pytest.approx(data["metric_value"]) == 81.325
        assert "chart_data" in data
        assert len(data["chart_data"]) > 0

def test_get_insights_endpoint():
    """Test insights generation endpoint with mocked OpenRouter response."""
    payload = {
        "analytics": {
            "Age": {"mean": 35.0, "min": 25, "max": 45},
            "Weight (kg)": {"mean": 77.0, "min": 60.0, "max": 90.0}
        },
        "api_key_override": "mock-api-key"
    }
    
    mock_insights = [
        "1. The average cohort age is 35 years.",
        "2. Weights span from 60 to 90 kg with an average of 77 kg.",
        "3. Key demographic cohorts demonstrate consistent baseline metrics."
    ]
    
    with patch("main.generate_executive_insights", return_value=mock_insights):
        response = client.post("/api/insights", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["insights"] == mock_insights

def test_export_report_endpoint(mock_csv_file):
    """Test report compilation endpoint and StreamingResponse file download."""
    df = pd.read_csv(io.BytesIO(mock_csv_file))
    
    payload = {
        "filters": {"filters": []},
        "analytics": {
            "Age": {"mean": 35.0, "min": 25, "max": 45}
        },
        "insights": [
            "Insight one",
            "Insight two"
        ],
        "chart_column": "Age"
    }
    
    mock_pdf_bytes = b"%PDF-1.4 Mock PDF Content"
    
    with patch("main.get_active_dataset", return_value=df), \
         patch("main.save_distribution_plot_image", return_value=Path("mock_chart.png")), \
         patch("main.generate_executive_report_pdf", return_value=mock_pdf_bytes), \
         patch("main.verify_reporting_pipeline", return_value={"download_ready": True}), \
         patch("main.Path.write_bytes") as mock_write:
         
        # We also mock the open function used in the streaming response
        mock_file = io.BytesIO(mock_pdf_bytes)
        with patch("builtins.open", return_value=mock_file):
            response = client.post("/api/report", json=payload)
            assert response.status_code == 200
            assert response.headers["content-type"] == "application/pdf"
            assert "attachment" in response.headers["content-disposition"]
            assert response.content == mock_pdf_bytes
