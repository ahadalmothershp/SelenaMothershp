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


def _extract_resources_from_reads(reads):
    """Pull individual FHIR resources out of the fhirAggregate.reads object."""
    resources = []
    resource_types_requested = set()
    resource_types_returned = set()

    patient = reads.get("patient")
    if patient and isinstance(patient, dict) and patient.get("resourceType"):
        resources.append(patient)
        resource_types_requested.add("Patient")
        resource_types_returned.add("Patient")

    search_keys = {
        "encounterSearch": "Encounter",
        "observationSearch": "Observation",
        "diagnosticReportSearch": "DiagnosticReport",
        "locationFollowUp": "Location",
    }

    for key, rt in search_keys.items():
        section = reads.get(key, {})
        resource_types_requested.add(rt)
        items = section.get("resources", [])
        if items:
            resource_types_returned.add(rt)
        for r in items:
            if isinstance(r, dict):
                resources.append(r)

    return resources, resource_types_requested, resource_types_returned


def _parse_all_resources(resources):
    """Run each resource through parse_resource; collect parsed clinical items."""
    return [parse_resource(r) for r in resources]


def _api_gw_response(status_code, body):
    return {
        "statusCode": status_code,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(body),
    }


def lambda_handler(event, context):
    """POST /baseline/analyze — accepts the other dev's fhirAggregate payload,
    runs parse + Bedrock/Claude, returns the agreed response schema."""

    try:
        if isinstance(event.get("body"), str):
            body = json.loads(event["body"])
        else:
            body = event

        fhir_agg = body.get("fhirAggregate", {})
        reads = fhir_agg.get("reads", {})
        run_context = fhir_agg.get("runContext", {})
        patient_id = run_context.get("patientId", "unknown")
        run_id = body.get("runId", run_context.get("runId"))

        log.info("baseline/analyze start — runId=%s patientId=%s", run_id, patient_id)

        resources, requested, returned = _extract_resources_from_reads(reads)
        missing = requested - returned

        clinical_items = _parse_all_resources(resources)
        meaningful_items = [
            item for item in clinical_items
            if item.get("clinical_unit") != "Unknown / Unsupported Resource"
        ]

        log.info(
            "Parsed %d resources, %d meaningful — requested=%s returned=%s missing=%s",
            len(resources), len(meaningful_items),
            sorted(requested), sorted(returned), sorted(missing),
        )

        baseline_output = None
        bedrock_info = {
            "invoked": False,
            "modelId": None,
            "inputTokens": None,
            "outputTokens": None,
            "latencyMs": None,
        }

        if meaningful_items:
            try:
                result = _invoke_bedrock(meaningful_items, patient_id)
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
                baseline_output = {"error": "Risk assessment temporarily unavailable"}
                bedrock_info["invoked"] = True
                bedrock_info["modelId"] = MODEL_ID
        elif resources:
            baseline_output = {
                "risk_level": "normal",
                "summary": "FHIR resources received but none produced structured clinical findings.",
                "key_findings": [],
                "risk_factors": [],
                "recommendation": "Expand resource types to include Condition for richer analysis.",
                "data_sufficiency": "limited",
            }

        response_body = {
            "schemaVersion": "phase0.syed.response.v1",
            "baselineOutput": baseline_output,
            "bedrock": bedrock_info,
            "fhirContextPatch": {
                "patientId": patient_id,
                "proofMode": "phase0-baseline",
                "resourcesRequested": sorted(requested),
                "resourcesReturned": sorted(returned),
                "resourcesMissing": sorted(missing),
            },
            "model": {
                "modelId": MODEL_ID if bedrock_info["invoked"] else None,
                "version": MODEL_VERSION if bedrock_info["invoked"] else None,
            },
        }

        log.info("baseline/analyze complete — runId=%s risk=%s bedrock_invoked=%s",
                 run_id,
                 baseline_output.get("risk_level") if isinstance(baseline_output, dict) else None,
                 bedrock_info["invoked"])

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
            "fhirContextPatch": None,
            "model": {"modelId": None, "version": None},
        })
