import json
from fastapi.testclient import TestClient
from analyzer.server import app

client = TestClient(app)

def test_read_root():
    """Verify that the home route serves the index.html page."""
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "LLM Training Data Quality Analyzer" in response.text

def test_api_analyze_json():
    """Verify JSON dataset analysis and report generation."""
    data = [
        {"text": "Hello world this is positive", "label": 1},
        {"text": "This is a short test", "label": 0}
    ]
    json_bytes = json.dumps(data).encode("utf-8")

    response = client.post(
        "/api/analyze",
        files={"file": ("test.json", json_bytes, "application/json")},
        data={
            "similarity_threshold": 0.9,
            "toxicity_threshold": 0.8,
            "min_token_threshold": 3,
            "gibberish_threshold": 0.5,
        }
    )
    assert response.status_code == 200
    res_json = response.json()
    assert res_json["success"] is True
    assert res_json["summary"]["total_records"] == 2
    assert "metrics" in res_json
    assert "records" in res_json
    assert len(res_json["records"]) == 2
    assert "report_json" in res_json
    assert "report_md" in res_json

def test_api_analyze_csv():
    """Verify CSV dataset analysis and report generation."""
    csv_data = "text,label\nHello world how are you today,1\nJust testing,0\n"
    csv_bytes = csv_data.encode("utf-8")

    response = client.post(
        "/api/analyze",
        files={"file": ("test.csv", csv_bytes, "text/csv")},
        data={
            "similarity_threshold": 0.9,
            "toxicity_threshold": 0.8,
            "min_token_threshold": 3,
            "gibberish_threshold": 0.5,
        }
    )
    assert response.status_code == 200
    res_json = response.json()
    assert res_json["success"] is True
    assert res_json["summary"]["total_records"] == 2
    assert res_json["records"][0]["fields"]["text"] == "Hello world how are you today"

def test_api_analyze_invalid_format():
    """Verify that unsupported extensions are handled fail-fast."""
    unsupported_data = b"some random content"
    response = client.post(
        "/api/analyze",
        files={"file": ("test.txt", unsupported_data, "text/plain")},
        data={
            "similarity_threshold": 0.9,
            "toxicity_threshold": 0.8,
            "min_token_threshold": 3,
            "gibberish_threshold": 0.5,
        }
    )
    # The ingestion engine treats unsupported extensions as skipped file in dir, 
    # but as a skipped file for single file it does not stream units. Let's verify.
    assert response.status_code == 400
    res_json = response.json()
    assert res_json["success"] is False
    assert "Unsupported file format" in res_json["error"]
