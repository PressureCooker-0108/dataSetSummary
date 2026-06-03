import os
import unittest
import pandas as pd
import logging
from pathlib import Path
import pytest
from unittest.mock import patch, MagicMock

from data_engine import (
    initialize_logger,
    load_csv,
    validate_dataset,
    analyze_schema,
    generate_schema_dictionary,
    generate_summary_report,
    extract_schema_metadata,
    validate_filter_payload,
    apply_filters,
    generate_analytics,
    generate_executive_insights,
    validate_insights,
    generate_executive_report_pdf,
    verify_reporting_pipeline,
    get_next_api_key,
    verify_openrouter_connection,
    recommend_visualizations
)

def test_initialize_logger():
    """Test logger initialization and logger instance properties."""
    logger = initialize_logger()
    assert isinstance(logger, logging.Logger)
    assert logger.name == "data_engine"

def test_validate_dataset_valid():
    """Test validate_dataset function with a valid DataFrame."""
    df = pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})
    assert validate_dataset(df) is True

def test_validate_dataset_invalid():
    """Test validate_dataset function with empty or invalid inputs."""
    assert validate_dataset(None) is False
    assert validate_dataset(pd.DataFrame()) is False
    assert validate_dataset(pd.DataFrame(columns=["a"])) is False  # 0 rows

def test_analyze_schema(tmp_path):
    """Test schema analysis generates the correct metadata structure."""
    df = pd.DataFrame({
        "A": [1, 2, None],
        "B": ["x", "y", "z"],
        "C": [1.1, 2.2, 3.3]
    })
    
    analysis = analyze_schema(df)
    
    assert analysis["dimensions"]["rows"] == 3
    assert analysis["dimensions"]["columns"] == 3
    assert analysis["duplicate_rows"] == 0
    assert "columns" in analysis
    
    assert analysis["columns"]["A"]["null_count"] == 1
    assert pytest.approx(analysis["columns"]["A"]["null_percentage"]) == 33.33333333333333
    assert analysis["columns"]["B"]["null_count"] == 0
    
    schema_dict = generate_schema_dictionary(df)
    assert schema_dict["A"].startswith("float")  # pandas upgrades int with Null to float
    assert schema_dict["B"] in ["object", "str", "string"]

def test_load_csv_nonexistent():
    """Test load_csv raises FileNotFoundError for nonexistent files."""
    with pytest.raises(FileNotFoundError):
        load_csv("nonexistent_file.csv")

def test_load_csv_and_generate_report(tmp_path):
    """Test loading a mock CSV and generating a ReportLab PDF report."""
    # Create temp CSV
    csv_file = tmp_path / "test_data.csv"
    df_original = pd.DataFrame({
        "Beneficiary_ID": [1, 2, 3],
        "District": ["District A", "District B", "District C"],
        "Amount": [100.5, 200.75, 150.0]
    })
    df_original.to_csv(csv_file, index=False)
    
    # Load
    df_loaded = load_csv(csv_file)
    assert df_loaded.shape == (3, 3)
    assert list(df_loaded.columns) == ["Beneficiary_ID", "District", "Amount"]
    
    # Report Generation
    report_pdf = tmp_path / "test_report.pdf"
    generated_path = generate_summary_report(df_loaded, output_path=report_pdf)
    
    assert os.path.exists(generated_path)
    assert generated_path == str(report_pdf.resolve())


# ==============================================================================
# CONVERSATIONAL QUERY ENGINE UNIT TESTS
# ==============================================================================

def test_extract_schema_metadata():
    """Test extract_schema_metadata computes ranges and cardinalities correctly."""
    df = pd.DataFrame({
        "Age": [20, 25, 30],
        "Gender": ["Male", "Female", "Male"],
        "Diet": ["Vegan", "Vegan", "Omnivore"],
        "Score": [1.5, 2.5, 3.5]
    })
    
    meta = extract_schema_metadata(df)
    assert "Age" in meta
    assert "Gender" in meta
    assert "Diet" in meta
    assert "Score" in meta
    
    # Age numeric range
    assert meta["Age"]["range"]["min"] == 20.0
    assert meta["Age"]["range"]["max"] == 30.0
    
    # Cardinality category list
    assert set(meta["Gender"]["unique_values"]) == {"Male", "Female"}
    assert meta["Gender"]["unique_count"] == 2


def test_validate_filter_payload_valid():
    """Test validate_filter_payload succeeds for correct payloads."""
    df = pd.DataFrame({
        "Age": [20, 25, 30],
        "Gender": ["Male", "Female", "Male"],
        "diet_type": ["vegan", "vegan", "omnivore"]
    })
    
    payload = {
        "filters": [
            {"column": "diet_type", "operator": "==", "value": "vegan"},
            {"column": "Age", "operator": "<", "value": 30},
            {"column": "Gender", "operator": "in", "value": ["Male", "Female"]}
        ]
    }
    
    assert validate_filter_payload(payload, df) is True


def test_validate_filter_payload_invalid():
    """Test validate_filter_payload fails for malformed or unsafe payloads."""
    df = pd.DataFrame({
        "Age": [20, 25, 30],
        "Gender": ["Male", "Female", "Male"]
    })
    
    # Test invalid column name
    bad_col = {
        "filters": [
            {"column": "nonexistent", "operator": "==", "value": 10}
        ]
    }
    assert validate_filter_payload(bad_col, df) is False
    
    # Test unsafe/non-whitelisted operator (eval statement attempt)
    bad_op = {
        "filters": [
            {"column": "Age", "operator": "eval", "value": "import os; os.system('echo unsafe')"}
        ]
    }
    assert validate_filter_payload(bad_op, df) is False
    
    # Test type incompatibility (numeric column vs string that cannot be cast)
    bad_type = {
        "filters": [
            {"column": "Age", "operator": "==", "value": "not-a-number"}
        ]
    }
    assert validate_filter_payload(bad_type, df) is False


def test_apply_filters():
    """Test apply_filters constructs boolean masks and filters data safely."""
    df = pd.DataFrame({
        "Age": [20, 25, 30, 35],
        "Gender": ["Male", "Female", "Male", "Female"],
        "Diet": ["Vegan", "Omnivore", "Vegan", "Omnivore"]
    })
    
    payload = {
        "filters": [
            {"column": "Diet", "operator": "==", "value": "Vegan"},
            {"column": "Age", "operator": "<", "value": 32}
        ]
    }
    
    filtered_df = apply_filters(df, payload)
    assert len(filtered_df) == 2
    assert list(filtered_df["Age"].tolist()) == [20, 30]


def test_generate_analytics():
    """Test generate_analytics returns correct aggregates and no raw row data."""
    df = pd.DataFrame({
        "Age": [20, 30, 40],
        "BMI": [22.0, 24.0, 26.0],
        "Gender": ["Male", "Female", "Male"]
    })
    
    analytics = generate_analytics(df)
    
    assert analytics["row_count"] == 3
    assert analytics["numeric_metrics"]["Age"]["mean"] == 30.0
    assert analytics["numeric_metrics"]["Age"]["min"] == 20.0
    assert analytics["numeric_metrics"]["Age"]["max"] == 40.0
    assert analytics["numeric_metrics"]["BMI"]["median"] == 24.0
    assert analytics["categorical_distributions"]["Gender"]["distribution"]["Male"] == 2
    
    # Ensure there is no raw dataframe records in the response keys
    assert "data" not in analytics
    assert "records" not in analytics
    assert "rows" not in analytics


# ==============================================================================
# REPORTING PIPELINE UNIT TESTS
# ==============================================================================

def test_generate_executive_insights_success():
    """Test generating executive insights using mocked OpenAI response."""
    analytics = {
        "row_count": 100,
        "numeric_metrics": {
            "BMI": {"mean": 23.5, "min": 18.0, "max": 30.0}
        }
    }
    
    mock_response = MagicMock()
    mock_response.choices[0].message.content = '{"insights": ["Insight 1", "Insight 2", "Insight 3"]}'
    
    with patch("openai.resources.chat.completions.Completions.create", return_value=mock_response):
        with patch("config.settings.settings.OPENROUTER_API_KEY", "dummy_key"):
            insights = generate_executive_insights(analytics)
            assert len(insights) == 3
            assert insights[0] == "Insight 1"


def test_validate_insights_formatting():
    """Test validation boundaries for executive insights lists."""
    assert validate_insights(["A", "B", "C"]) is True
    # Incomplete list length
    assert validate_insights(["A", "B"]) is False
    # Empty element
    assert validate_insights(["A", "", "C"]) is False
    # Markdown tags present
    assert validate_insights(["A", "**B**", "C"]) is False


def test_generate_executive_report_pdf_bytes():
    """Test PDF generation builds a populated byte stream."""
    analytics = {
        "row_count": 50,
        "numeric_metrics": {
            "BMI": {"mean": 24.2},
            "Calories": {"mean": 2100.0}
        }
    }
    insights = ["Insight 1", "Insight 2", "Insight 3"]
    
    # Run PDF rendering without a chart image
    pdf_bytes = generate_executive_report_pdf(
        filters={"filters": []},
        analytics_summary=analytics,
        insights=insights,
        chart_path=""
    )
    
    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 0


def test_verify_reporting_pipeline_status():
    """Test verify_reporting_pipeline executes self-diagnostics correctly."""
    result = verify_reporting_pipeline()
    assert result["insights_generated"] is True
    assert result["pdf_generated"] is True
    assert result["download_ready"] is True


def test_api_key_rotation_settings():
    """Test get_next_api_key cycles sequentially through settings candidates."""
    with patch("config.settings.settings.OPENROUTER_API_KEY", "key_main"):
        with patch("config.settings.settings.OPENROUTER_API_KEY_1", "key_1"):
            with patch("config.settings.settings.OPENROUTER_API_KEY_2", "key_2"):
                with patch("config.settings.settings.OPENROUTER_API_KEY_3", ""):
                    with patch("config.settings.settings.OPENROUTER_API_KEY_4", "key_4"):
                        # Get a sequence of rotated keys
                        k1 = get_next_api_key()
                        k2 = get_next_api_key()
                        k3 = get_next_api_key()
                        k4 = get_next_api_key()
                        k5 = get_next_api_key() # Should wrap around
                        
                        # Set of active keys: key_main, key_1, key_2, key_4 (key_3 is empty)
                        expected = ["key_main", "key_1", "key_2", "key_4"]
                        assert k1 in expected
                        assert k2 in expected
                        assert k3 in expected
                        assert k4 in expected
                        assert k5 == k1


def test_api_key_rotation_overrides():
    """Test get_next_api_key cycles through comma-separated overrides."""
    overrides = "keyA,  keyB, keyC  "
    
    k1 = get_next_api_key(overrides)
    k2 = get_next_api_key(overrides)
    k3 = get_next_api_key(overrides)
    k4 = get_next_api_key(overrides)
    
    assert k1 == "keyA"
    assert k2 == "keyB"
    assert k3 == "keyC"
    assert k4 == "keyA" # wraparound


def test_verify_openrouter_connection_success():
    """Test verify_openrouter_connection returning true on valid response."""
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "OK"
    
    with patch("openai.resources.chat.completions.Completions.create", return_value=mock_response):
        res = verify_openrouter_connection("valid_key")
        assert res is True


def test_verify_openrouter_connection_failure():
    """Test verify_openrouter_connection returning false on error."""
    with patch("openai.resources.chat.completions.Completions.create", side_effect=Exception("API Error")):
        res = verify_openrouter_connection("invalid_key")
        assert res is False


def test_execute_with_backoff_retry_then_success():
    """Test execute_with_backoff retries on 429 and eventually succeeds."""
    from openai import RateLimitError
    from data_engine import execute_with_backoff
    
    call_count = 0
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "Success"
    
    def mock_api_call():
        nonlocal call_count
        call_count += 1
        if call_count < 2:
            # Raise RateLimitError on first attempt
            # RateLimitError requires message, response, and body parameters in latest version
            raise RateLimitError(
                message="Too Many Requests",
                response=MagicMock(status_code=429),
                body=None
            )
        return mock_response
        
    with patch("time.sleep") as mock_sleep:  # Speed up test by not sleeping
        result = execute_with_backoff(mock_api_call)
        assert result == mock_response
        assert call_count == 2
        mock_sleep.assert_called_once_with(2.0)


def test_recommend_visualizations_fallback():
    """Test recommend_visualizations returns deterministic fallback when API key is missing or offline."""
    # Build a simple schema metadata structure
    schema_meta = {
        "Age": {"dtype": "int64", "range": {"min": 18, "max": 80}},
        "BMI": {"dtype": "float64", "range": {"min": 15.0, "max": 40.0}},
        "Gender": {"dtype": "object", "unique_count": 2, "unique_values": ["Male", "Female"]},
        "Session_Duration (hours)": {"dtype": "float64", "range": {"min": 0.5, "max": 4.0}},
        "Calories_Burned": {"dtype": "float64", "range": {"min": 100, "max": 1000}}
    }
    
    # Run with empty api_key to force fallback
    recs = recommend_visualizations(schema_meta, api_key="")
    assert "recommendations" in recs
    assert len(recs["recommendations"]) == 3
    
    types = [r["type"] for r in recs["recommendations"]]
    assert "distribution" in types
    assert "boxplot" in types
    assert "correlation" in types


def test_recommend_visualizations_success():
    """Test recommend_visualizations successfully parses valid JSON response from OpenAI/OpenRouter API."""
    schema_meta = {
        "Age": {"dtype": "int64", "range": {"min": 18, "max": 80}},
        "Gender": {"dtype": "object", "unique_count": 2, "unique_values": ["Male", "Female"]}
    }
    
    mock_response = MagicMock()
    mock_response.choices[0].message.content = """
    {
      "recommendations": [
        {
          "type": "distribution",
          "title": "LLM Distribution of Age",
          "description": "Valuable distribution description",
          "x_column": "Age",
          "y_column": null,
          "hue_column": null
        },
        {
          "type": "boxplot",
          "title": "LLM Boxplot",
          "description": "Valuable boxplot description",
          "x_column": "Gender",
          "y_column": "Age",
          "hue_column": null
        },
        {
          "type": "correlation",
          "title": "LLM Correlation",
          "description": "Valuable correlation description",
          "x_column": "Age",
          "y_column": "Age",
          "hue_column": "Gender"
        }
      ]
    }
    """
    
    with patch("openai.resources.chat.completions.Completions.create", return_value=mock_response):
        # We pass a non-empty api_key override or mock settings so that it invokes the API
        with patch("data_engine.get_next_api_key", return_value="mock_key"):
            recs = recommend_visualizations(schema_meta, api_key="mock_key")
            assert "recommendations" in recs
            assert len(recs["recommendations"]) == 3
            assert recs["recommendations"][0]["title"] == "LLM Distribution of Age"
            assert recs["recommendations"][1]["type"] == "boxplot"
            assert recs["recommendations"][2]["hue_column"] == "Gender"


def test_recommend_visualizations_malformed_json_fallback():
    """Test recommend_visualizations falls back to deterministic recommendations if LLM returns malformed JSON."""
    schema_meta = {
        "Age": {"dtype": "int64", "range": {"min": 18, "max": 80}},
        "Gender": {"dtype": "object", "unique_count": 2, "unique_values": ["Male", "Female"]}
    }
    
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "not a valid json string"
    
    with patch("openai.resources.chat.completions.Completions.create", return_value=mock_response):
        with patch("data_engine.get_next_api_key", return_value="mock_key"):
            recs = recommend_visualizations(schema_meta, api_key="mock_key")
            assert "recommendations" in recs
            assert len(recs["recommendations"]) == 3
            # Check that it falls back to the deterministic format
            assert recs["recommendations"][0]["title"].startswith("Distribution of")
