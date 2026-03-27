import json
import logging
import os
import re
import time
from typing import Any

import boto3
from botocore.config import Config

from lambda_function import parse_resource

log = logging.getLogger("selena.analyze")
log.setLevel(logging.INFO)

BEDROCK_REGION = os.getenv("AWS_DEFAULT_REGION", "us-west-2")
MODEL_ID = "us.anthropic.claude-sonnet-4-6"
MODEL_VERSION = "phase0-baseline-v2"
BEDROCK_TIMEOUT = int(os.getenv("BEDROCK_TIMEOUT_SECONDS", 30))

PHASE0_AI_SCHEMA = "phase0.ai.request.v1"
PHASE0_FHIR_SCHEMA = "phase0.fhir.case.v1"

PHASE0_RESOURCE_FAMILIES = [
    "patient",              # special: object or None
    "allergies",
    "conditions",
    "diagnosticReports",
    "encounters",
    "locations",
    "medicationRequests",
    "procedures",
    "vitals",
]

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
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
        log.error("Claude returned JSON but not an object: %s", type(parsed).__name__)
        return {"error": "Model returned JSON in an unexpected shape"}
    except json.JSONDecodeError:
        log.error("Claude returned non-JSON output: %s", raw_text[:500])
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
# Request parsing
# ---------------------------------------------------------------------------

def _extract_body(event: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(event, dict):
        raise ValueError("Event must be a JSON object")

    if "body" not in event:
        return event

    raw_body = event["body"]

    if isinstance(raw_body, str):
        return json.loads(raw_body)

    if isinstance(raw_body, dict):
        return raw_body

    raise ValueError("Request body must be a JSON object or JSON string")


# ---------------------------------------------------------------------------
# Resource extraction
# Supports:
#   1) New phase0.ai.request.v1 -> fhirAggregate.resources
#   2) Legacy v3 -> executionPayloads
#   3) Legacy v1 -> reads
# ---------------------------------------------------------------------------

def _extract_resources_phase0_case(fhir_agg: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Extract resources from the new Phase 0 single-case payload shape:

    fhirAggregate = {
      "schemaVersion": "phase0.fhir.case.v1",
      "resources": {
        "patient": {...} | None,
        "allergies": [...],
        "conditions": [...],
        "diagnosticReports": [...],
        "encounters": [...],
        "locations": [...],
        "medicationRequests": [...],
        "procedures": [...],
        "vitals": [...],
      }
    }
    """
    resources_block = fhir_agg.get("resources", {})
    if not isinstance(resources_block, dict):
        return []

    extracted = []

    # patient is a single object or None
    patient = resources_block.get("patient")
    if isinstance(patient, dict) and patient.get("resourceType"):
        extracted.append(patient)

    # all other families are lists
    for family in PHASE0_RESOURCE_FAMILIES:
        if family == "patient":
            continue

        family_items = resources_block.get(family, [])
        if not isinstance(family_items, list):
            log.warning("Expected list for resources.%s but got %s", family, type(family_items).__name__)
            continue

        for item in family_items:
            if isinstance(item, dict) and item.get("resourceType"):
                extracted.append(item)
            else:
                log.debug("Skipping invalid resource in family=%s", family)

    return extracted


def _extract_resources_v3(fhir_agg: dict[str, Any]) -> list[dict[str, Any]]:
    """Legacy: extract resources from the v3 executionPayloads structure."""
    payloads = fhir_agg.get("executionPayloads", {})
    resources = []

    if not isinstance(payloads, dict):
        return resources

    for _, payload in payloads.items():
        if not isinstance(payload, dict):
            continue
        for r in payload.get("resources", []):
            if isinstance(r, dict) and r.get("resourceType"):
                resources.append(r)

    return resources


def _extract_resources_v1(fhir_agg: dict[str, Any]) -> list[dict[str, Any]]:
    """Legacy: extract resources from the v1 reads structure."""
    reads = fhir_agg.get("reads", {})
    resources = []

    if not isinstance(reads, dict):
        return resources

    patient = reads.get("patient")
    if isinstance(patient, dict) and patient.get("resourceType"):
        resources.append(patient)

    for key in [
        "encounterSearch",
        "observationSearch",
        "diagnosticReportSearch",
        "locationFollowUp",
        "conditionSearch",
        "procedureSearch",
        "allergyIntoleranceSearch",
        "medicationRequestRead",
    ]:
        section = reads.get(key, {})
        if not isinstance(section, dict):
            continue

        for r in section.get("resources", []):
            if isinstance(r, dict) and r.get("resourceType"):
                resources.append(r)

    return resources


def _extract_resources(body: dict[str, Any], fhir_agg: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Choose the correct extraction strategy.

    Priority:
      1) New Phase 0 request contract
      2) Legacy executionPayloads
      3) Legacy reads
    """
    if (
        body.get("schemaVersion") == PHASE0_AI_SCHEMA
        and fhir_agg.get("schemaVersion") == PHASE0_FHIR_SCHEMA
        and isinstance(fhir_agg.get("resources"), dict)
    ):
        return _extract_resources_phase0_case(fhir_agg)

    if fhir_agg.get("executionPayloads"):
        return _extract_resources_v3(fhir_agg)

    if fhir_agg.get("reads"):
        return _extract_resources_v1(fhir_agg)

    return []


# ---------------------------------------------------------------------------
# Parse all resources
# ---------------------------------------------------------------------------

def _parse_all_resources(resources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    parsed = []

    for r in resources:
        try:
            parsed.append(parse_resource(r))
        except Exception:
            log.exception(
                "Failed to parse resource resourceType=%s id=%s",
                r.get("resourceType"),
                r.get("id"),
            )
            parsed.append({
                "resourceType": r.get("resourceType"),
                "id": r.get("id"),
                "parse_error": True,
            })

    return parsed


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
        body = _extract_body(event)

        fhir_agg = body.get("fhirAggregate", {})
        if not isinstance(fhir_agg, dict):
            fhir_agg = {}

        run_context = fhir_agg.get("runContext", {})
        if not isinstance(run_context, dict):
            run_context = {}

        patient_id = run_context.get("patientId", "unknown")
        run_id = body.get("runId", run_context.get("runId", "unknown"))

        log.info(
            "baseline/analyze start — runId=%s patientId=%s bodySchema=%s fhirSchema=%s",
            run_id,
            patient_id,
            body.get("schemaVersion"),
            fhir_agg.get("schemaVersion"),
        )

        resources = _extract_resources(body, fhir_agg)
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
                baseline_output = {
                    "error": "Risk assessment temporarily unavailable",
                    "summary": "Model invocation failed",
                    "data_sufficiency": "limited",
                }
                bedrock_info["invoked"] = True
                bedrock_info["modelId"] = MODEL_ID
        else:
            baseline_output = {
                "error": "No FHIR resources found in the payload",
                "summary": "No parseable clinical data was provided",
                "data_sufficiency": "limited",
            }

        response_body = {
            "schemaVersion": "phase0.ai.response.v1",
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

    except json.JSONDecodeError:
        log.warning("Received non-JSON request body")
        return _api_gw_response(400, {
            "schemaVersion": "phase0.ai.response.v1",
            "error": "Request body must be valid JSON",
            "baselineOutput": None,
            "bedrock": {
                "invoked": False,
                "modelId": None,
                "inputTokens": None,
                "outputTokens": None,
                "latencyMs": None,
            },
        })
    except ValueError as exc:
        log.warning("Invalid request format: %s", exc)
        return _api_gw_response(400, {
            "schemaVersion": "phase0.ai.response.v1",
            "error": str(exc),
            "baselineOutput": None,
            "bedrock": {
                "invoked": False,
                "modelId": None,
                "inputTokens": None,
                "outputTokens": None,
                "latencyMs": None,
            },
        })
    except Exception:
        log.exception("Unhandled error in baseline/analyze")
        return _api_gw_response(500, {
            "schemaVersion": "phase0.ai.response.v1",
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
