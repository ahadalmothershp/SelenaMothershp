import json
import logging
import os
import re
import time

import boto3
from botocore.config import Config

from lambda_function import parse_resource

log = logging.getLogger("selena.analyze")

BEDROCK_REGION = os.getenv("AWS_DEFAULT_REGION", "us-west-2")
MODEL_ID = "us.anthropic.claude-sonnet-4-6"
MODEL_VERSION = "phase0-baseline-v1"
BEDROCK_TIMEOUT = int(os.getenv("BEDROCK_TIMEOUT_SECONDS", 30))

bedrock = boto3.client(
    "bedrock-runtime",
    region_name=BEDROCK_REGION,
    config=Config(
        retries={"max_attempts": 2, "mode": "adaptive"},
        read_timeout=BEDROCK_TIMEOUT,
        connect_timeout=5,
    ),
)


# ---------------------------------------------------------------------------
# Prompt & Bedrock
# ---------------------------------------------------------------------------

def _build_prompt(clinical_data, patient_id):
    return f"""You are a healthcare risk assessment assistant.

Analyze the following clinical data for patient {patient_id} and return JSON only.

Return format:
{{
  "risk_level": "normal|low|medium|high",
  "summary": "short summary",
  "key_findings": [],
  "risk_factors": [],
  "recommendation": "short cautious recommendation",
  "data_sufficiency": "limited|moderate|good"
}}

Rules:
- Do not diagnose.
- Use only the provided data.
- If data is limited, say so.
- Be conservative and safe.
- Return JSON only.

Clinical data:
{json.dumps(clinical_data, indent=2)}
"""


def _parse_claude_output(raw_text):
    cleaned = re.sub(r"^```json\s*|\s*```$", "", raw_text.strip(), flags=re.MULTILINE)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        log.error("Claude returned non-JSON output: %s", raw_text[:200])
        return {"error": "Model did not return valid JSON"}


def _invoke_bedrock(clinical_data, patient_id):
    prompt = _build_prompt(clinical_data, patient_id)
    start = time.time()

    log.info("Invoking Bedrock model=%s for patient=%s", MODEL_ID, patient_id)

    response = bedrock.converse(
        modelId=MODEL_ID,
        messages=[{"role": "user", "content": [{"text": prompt}]}],
        inferenceConfig={"maxTokens": 6400, "temperature": 0.1},
    )

    latency_ms = int((time.time() - start) * 1000)
    text_output = response["output"]["message"]["content"][0]["text"]
    usage = response.get("usage", {})

    log.info(
        "Bedrock responded in %dms — input=%d output=%d tokens",
        latency_ms,
        usage.get("inputTokens", 0),
        usage.get("outputTokens", 0),
    )

    return {
        "parsed": _parse_claude_output(text_output),
        "input_tokens": usage.get("inputTokens"),
        "output_tokens": usage.get("outputTokens"),
        "latency_ms": latency_ms,
    }


# ---------------------------------------------------------------------------
# Resource extraction — supports both v3 (executionPayloads) and v1 (reads)
# ---------------------------------------------------------------------------

def _extract_resources_v3(fhir_agg):
    """Extract resources from the v3 executionPayloads structure."""
    payloads = fhir_agg.get("executionPayloads", {})
    resources = []

    for label, payload in payloads.items():
        for r in payload.get("resources", []):
            if isinstance(r, dict) and r.get("resourceType"):
                resources.append(r)

    return resources


def _extract_resources_v1(fhir_agg):
    """Extract resources from the v1 reads structure (backward compat)."""
    reads = fhir_agg.get("reads", {})
    resources = []

    patient = reads.get("patient")
    if patient and isinstance(patient, dict) and patient.get("resourceType"):
        resources.append(patient)

    for key in ["encounterSearch", "observationSearch", "diagnosticReportSearch",
                "locationFollowUp", "conditionSearch", "procedureSearch",
                "allergyIntoleranceSearch", "medicationRequestRead"]:
        section = reads.get(key, {})
        for r in section.get("resources", []):
            if isinstance(r, dict):
                resources.append(r)

    return resources


def _extract_resources(fhir_agg):
    """Pick the right extraction method based on schema version."""
    schema = fhir_agg.get("schemaVersion", "")

    if fhir_agg.get("executionPayloads"):
        return _extract_resources_v3(fhir_agg)
    if fhir_agg.get("reads"):
        return _extract_resources_v1(fhir_agg)

    return []


# ---------------------------------------------------------------------------
# Parse all resources
# ---------------------------------------------------------------------------

def _parse_all_resources(resources):
    return [parse_resource(r) for r in resources]


# ---------------------------------------------------------------------------
# API Gateway response helper
# ---------------------------------------------------------------------------

def _api_gw_response(status_code, body):
    return {
        "statusCode": status_code,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(body),
    }


# ---------------------------------------------------------------------------
# Lambda handler — POST /baseline/analyze
# ---------------------------------------------------------------------------

def lambda_handler(event, context):
    try:
        if isinstance(event.get("body"), str):
            body = json.loads(event["body"])
        else:
            body = event

        fhir_agg = body.get("fhirAggregate", {})
        run_context = fhir_agg.get("runContext", {})
        patient_id = run_context.get("patientId", "unknown")
        run_id = body.get("runId", run_context.get("runId"))

        log.info("baseline/analyze start — runId=%s patientId=%s", run_id, patient_id)

        resources = _extract_resources(fhir_agg)
        parsed_items = _parse_all_resources(resources)

        log.info(
            "Extracted %d raw resources, parsed %d items",
            len(resources), len(parsed_items),
        )

        baseline_output = None
        bedrock_info = {
            "invoked": False,
            "modelId": None,
            "inputTokens": None,
            "outputTokens": None,
            "latencyMs": None,
        }

        if parsed_items:
            try:
                result = _invoke_bedrock(parsed_items, patient_id)
                baseline_output = result["parsed"]
                bedrock_info = {
                    "invoked": True,
                    "modelId": MODEL_ID,
                    "inputTokens": result["input_tokens"],
                    "outputTokens": result["output_tokens"],
                    "latencyMs": result["latency_ms"],
                }
            except Exception:
                log.exception("Bedrock invocation failed for patient=%s", patient_id)
                baseline_output = "Risk assessment temporarily unavailable"
                bedrock_info["invoked"] = True
                bedrock_info["modelId"] = MODEL_ID
        else:
            baseline_output = "No FHIR resources found in the payload"

        response_body = {
            "schemaVersion": "phase0.syed.response.v1",
            "baselineOutput": baseline_output,
            "bedrock": bedrock_info,
        }

        if bedrock_info["invoked"]:
            response_body["model"] = {
                "modelId": MODEL_ID,
                "version": MODEL_VERSION,
            }

        log.info(
            "baseline/analyze complete — runId=%s risk=%s bedrock_invoked=%s",
            run_id,
            baseline_output.get("risk_level") if isinstance(baseline_output, dict) else None,
            bedrock_info["invoked"],
        )

        return _api_gw_response(200, response_body)

    except Exception:
        log.exception("Unhandled error in baseline/analyze")
        return _api_gw_response(500, {
            "schemaVersion": "phase0.syed.response.v1",
            "error": "Internal server error",
            "baselineOutput": None,
            "bedrock": {
                "invoked": False,
                "modelId": None,
                "inputTokens": None,
                "outputTokens": None,
                "latencyMs": None,
            },
        })
