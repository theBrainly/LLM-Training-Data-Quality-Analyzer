import os
import shutil
import uuid
import json
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from analyzer.ingestion import IngestionEngine, IngestionConfig
from analyzer.parsers import Parser
from analyzer.combine import combine
from analyzer.pipeline import analyze, PipelineConfig
from analyzer.detectors.quality import QualityConfig
from analyzer.detectors.pii import PIIDetector
from analyzer.report import ReportGenerator, OutputFormat
from analyzer.models import Schema, FieldSpec, FieldType, IssueCategory

app = FastAPI(
    title="LLM Training Data Quality Analyzer",
    description="Interactive Web UI to inspect datasets and view quality metrics."
)

# Create a temporary upload directory in the workspace
WORKSPACE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPLOAD_DIR = os.path.join(WORKSPACE_DIR, "temp_uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Helper to serialize record location
def _location_to_dict(location) -> dict | None:
    if location is None:
        return None
    return {
        "source_file": os.path.basename(location.source_file) if location.source_file else "",
        "line_number": location.line_number,
        "array_index": location.array_index,
        "row_group": location.row_group,
        "row_index": location.row_index,
    }

# Helper to serialize quality issue
def _issue_to_dict(issue) -> dict:
    return {
        "category": issue.category.value,
        "location": _location_to_dict(issue.location),
        "field_name": issue.field_name,
        "related_location": _location_to_dict(issue.related_location),
        "detail": issue.detail,
        "pii_category": issue.pii_category,
        "span": {
            "start": issue.span.start,
            "end": issue.span.end
        } if issue.span else None,
        "score": issue.score,
    }

# Helper to match issues to a specific record
def _is_issue_for_record(issue, record) -> bool:
    if issue.location is None:
        return False
    loc = issue.location
    r_loc = record.location
    if loc.source_file != r_loc.source_file:
        return False
    if loc.array_index is not None and loc.array_index == r_loc.array_index:
        return True
    if loc.line_number is not None and loc.line_number == r_loc.line_number:
        return True
    if loc.row_index is not None and loc.row_index == r_loc.row_index and loc.row_group == r_loc.row_group:
        return True
    return False

@app.post("/api/analyze")
async def api_analyze(
    file: UploadFile = File(...),
    similarity_threshold: float = Form(0.9),
    toxicity_threshold: float = Form(0.8),
    min_token_threshold: int = Form(3),
    gibberish_threshold: float = Form(0.5),
    schema_json: Optional[str] = Form(None),
):
    # Create a unique file name
    file_id = str(uuid.uuid4())
    _, ext = os.path.splitext(file.filename)
    temp_file_name = f"{file_id}{ext}"
    temp_file_path = os.path.join(UPLOAD_DIR, temp_file_name)

    try:
        # Save file to temp path
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 1. Ingest
        engine = IngestionEngine()
        ingestion_result = engine.ingest(temp_file_path, IngestionConfig())

        if len(ingestion_result.skipped_files) > 0:
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "error": "Unsupported file format. Please upload JSON, JSONL, CSV, or Parquet files."
                }
            )

        if ingestion_result.error is not None:
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "error": f"Ingestion error: {ingestion_result.error.__class__.__name__} - {str(ingestion_result.error)}"
                }
            )

        # 2. Parse & Combine
        parser = Parser()
        combined = combine(ingestion_result, parser)
        dataset = combined.dataset
        parse_issues = combined.issues

        # Handle optional custom schema
        schema = None
        if schema_json:
            try:
                schema_data = json.loads(schema_json)
                fields = []
                for field_data in schema_data.get("fields", []):
                    fields.append(FieldSpec(
                        name=field_data["name"],
                        type=FieldType(field_data["type"]),
                        required=field_data.get("required", False)
                    ))
                schema = Schema(fields=fields)
            except Exception as e:
                return JSONResponse(
                    status_code=400,
                    content={
                        "success": False,
                        "error": f"Invalid schema format: {str(e)}"
                    }
                )

        # 3. Analyze Pipeline
        pipeline_config = PipelineConfig(
            similarity_threshold=similarity_threshold,
            toxicity_threshold=toxicity_threshold,
            quality_config=QualityConfig(
                min_token_threshold=min_token_threshold,
                gibberish_threshold=gibberish_threshold
            ),
            schema=schema
        )

        analysis_result = analyze(dataset, config=pipeline_config, parse_issues=parse_issues)

        # 4. Generate Reports
        report_gen = ReportGenerator()
        report = report_gen.build(dataset, analysis_result.metrics, analysis_result.issues)

        json_serialize_result = report_gen.serialize(report, OutputFormat.JSON)
        md_serialize_result = report_gen.serialize(report, OutputFormat.MARKDOWN)

        # 5. Process records and associate PII redactions/issues
        pii_detector = PIIDetector()
        processed_records = []
        for i, record in enumerate(dataset.records):
            # Redact if PII is present in the issues for this record
            record_issues = [iss for iss in analysis_result.issues if _is_issue_for_record(iss, record)]
            has_pii = any(iss.category == IssueCategory.PII for iss in record_issues)
            redacted_fields = None
            if has_pii:
                redacted_rec = pii_detector.redact(record)
                redacted_fields = redacted_rec.fields

            processed_records.append({
                "index": i,
                "fields": record.fields,
                "redacted_fields": redacted_fields,
                "location": _location_to_dict(record.location),
                "issues": [_issue_to_dict(iss) for iss in record_issues]
            })

        # Calculate category distribution for charts
        category_counts = {cat.value: count for cat, count in report.category_counts.items()}

        return {
            "success": True,
            "filename": file.filename,
            "metrics": {
                "record_count": report.metrics.record_count,
                "mean_tokens": report.metrics.mean_tokens,
                "min_tokens": report.metrics.min_tokens,
                "max_tokens": report.metrics.max_tokens,
                "issue_record_proportion": report.metrics.issue_record_proportion,
                "quality_score": report.metrics.quality_score,
            },
            "summary": {
                "total_records": report.total_records,
                "total_issues": report.total_issues,
            },
            "category_counts": category_counts,
            "issues": [_issue_to_dict(iss) for iss in analysis_result.issues],
            "records": processed_records,
            "report_json": json_serialize_result.text,
            "report_md": md_serialize_result.text,
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": f"Internal server error during analysis: {str(e)}"
            }
        )
    finally:
        # Clean up temporary uploaded file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

# Serve the static UI files
@app.get("/")
def read_root():
    return FileResponse(os.path.join(os.path.dirname(__file__), "static", "index.html"))

# Mount the static files directory
app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static")), name="static")
