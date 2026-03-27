import json
import logging
import os
from typing import Any

log = logging.getLogger("selena.core")
log.setLevel(logging.INFO)

API_KEY = os.environ.get("API_KEY")

PHASE0_AI_SCHEMA = "phase0.ai.request.v1"
PHASE0_FHIR_SCHEMA = "phase0.fhir.case.v1"

REQUIRED_TOP_LEVEL_KEYS = {
    "authContext",
    "fhirAggregate",
    "runId",
    "runtimeContext",
    "schemaVersion",
    "timestamp",
}

REQUIRED_RESOURCE_FAMILIES = [
    "allergies",
    "conditions",
    "diagnosticReports",
    "encounters",
    "locations",
    "medicationRequests",
    "patient",
    "procedures",
    "vitals",
]


# ---------------------------------------------------------------------------
# FHIR Resource Parsers (legacy single-resource mode)
# ---------------------------------------------------------------------------

def parse_condition(resource):
    code = resource.get("code", {})
    coding = code.get("coding", [])

    snomed, icd9, icd10 = None, None, None
    for c in coding:
        system = c.get("system")
        if system == "http://snomed.info/sct":
            snomed = {"code": c.get("code"), "display": c.get("display")}
        elif system == "http://hl7.org/fhir/sid/icd-9-cm":
            icd9 = {"code": c.get("code"), "display": c.get("display")}
        elif system == "http://hl7.org/fhir/sid/icd-10-cm":
            icd10 = {"code": c.get("code"), "display": c.get("display")}

    categories = [cat.get("text") for cat in resource.get("category", []) if cat.get("text")]

    return {
        "resourceType": "Condition",
        "id": resource.get("id"),
        "display": code.get("text"),
        "categories": categories,
        "snomed": snomed,
        "icd9": icd9,
        "icd10": icd10,
        "clinicalStatus": _extract_codeable_text(resource.get("clinicalStatus")),
        "verificationStatus": _extract_codeable_text(resource.get("verificationStatus")),
        "onsetDateTime": resource.get("onsetDateTime"),
        "abatementDateTime": resource.get("abatementDateTime"),
        "recordedDate": resource.get("recordedDate"),
        "subject_reference": resource.get("subject", {}).get("reference"),
    }


def parse_patient(resource):
    names = resource.get("name", [])
    name_str = None
    if names:
        n = names[0]
        given = " ".join(n.get("given", []))
        family = n.get("family", "")
        name_str = f"{given} {family}".strip()

    return {
        "resourceType": "Patient",
        "id": resource.get("id"),
        "name": name_str,
        "gender": resource.get("gender"),
        "birthDate": resource.get("birthDate"),
        "active": resource.get("active"),
    }


def parse_encounter(resource):
    enc_class = resource.get("class", {})
    types = []
    for t in resource.get("type", []):
        for c in t.get("coding", []):
            types.append(c.get("display"))

    period = resource.get("period", {})

    return {
        "resourceType": "Encounter",
        "id": resource.get("id"),
        "status": resource.get("status"),
        "class": enc_class.get("display") or enc_class.get("code"),
        "types": [t for t in types if t],
        "periodStart": period.get("start"),
        "periodEnd": period.get("end"),
        "location_reference": _extract_location_ref(resource),
    }


def parse_observation(resource):
    code = resource.get("code", {})
    code_display = code.get("text") or _first_coding_display(code)

    value = None
    unit = None
    vq = resource.get("valueQuantity")
    if vq:
        value = vq.get("value")
        unit = vq.get("unit")
    elif resource.get("valueString"):
        value = resource.get("valueString")
    elif resource.get("valueCodeableConcept"):
        value = _extract_codeable_text(resource.get("valueCodeableConcept"))

    components = []
    for comp in resource.get("component", []):
        comp_code = comp.get("code", {})
        comp_vq = comp.get("valueQuantity", {})
        components.append({
            "name": comp_code.get("text") or _first_coding_display(comp_code),
            "value": comp_vq.get("value"),
            "unit": comp_vq.get("unit"),
        })

    interpretations = []
    for interp in resource.get("interpretation", []):
        for c in interp.get("coding", []):
            interpretations.append(c.get("display") or c.get("code"))

    return {
        "resourceType": "Observation",
        "id": resource.get("id"),
        "status": resource.get("status"),
        "display": code_display,
        "value": value,
        "unit": unit,
        "components": components if components else None,
        "interpretation": interpretations if interpretations else None,
        "effectiveDateTime": resource.get("effectiveDateTime"),
        "category": _extract_obs_category(resource),
    }


def parse_medication_request(resource):
    med_code = resource.get("medicationCodeableConcept", {})
    med_ref = resource.get("medicationReference", {})

    dosage_instructions = []
    for d in resource.get("dosageInstruction", []):
        dosage_instructions.append(d.get("text") or d.get("patientInstruction"))

    return {
        "resourceType": "MedicationRequest",
        "id": resource.get("id"),
        "status": resource.get("status"),
        "intent": resource.get("intent"),
        "medication": med_code.get("text") or _first_coding_display(med_code) or med_ref.get("display"),
        "dosageInstructions": [d for d in dosage_instructions if d],
        "authoredOn": resource.get("authoredOn"),
        "requester_display": resource.get("requester", {}).get("display"),
    }


def parse_procedure(resource):
    code = resource.get("code", {})
    return {
        "resourceType": "Procedure",
        "id": resource.get("id"),
        "status": resource.get("status"),
        "display": code.get("text") or _first_coding_display(code),
        "performedDateTime": resource.get("performedDateTime"),
        "performedPeriod": resource.get("performedPeriod"),
    }


def parse_diagnostic_report(resource):
    code = resource.get("code", {})
    results = []
    for ref in resource.get("result", []):
        results.append(ref.get("display") or ref.get("reference"))

    return {
        "resourceType": "DiagnosticReport",
        "id": resource.get("id"),
        "status": resource.get("status"),
        "display": code.get("text") or _first_coding_display(code),
        "category": _extract_dr_category(resource),
        "effectiveDateTime": resource.get("effectiveDateTime"),
        "issued": resource.get("issued"),
        "results": [r for r in results if r] if results else None,
    }


def parse_allergy_intolerance(resource):
    code = resource.get("code", {})

    reactions = []
    for r in resource.get("reaction", []):
        manifestations = []
        for m in r.get("manifestation", []):
            manifestations.append(m.get("text") or _first_coding_display(m))
        reactions.append({
            "substance": _extract_codeable_text(r.get("substance")),
            "manifestations": [m for m in manifestations if m],
            "severity": r.get("severity"),
        })

    return {
        "resourceType": "AllergyIntolerance",
        "id": resource.get("id"),
        "clinicalStatus": _extract_codeable_text(resource.get("clinicalStatus")),
        "verificationStatus": _extract_codeable_text(resource.get("verificationStatus")),
        "display": code.get("text") or _first_coding_display(code),
        "category": resource.get("category"),
        "criticality": resource.get("criticality"),
        "reactions": reactions if reactions else None,
        "recordedDate": resource.get("recordedDate"),
    }


def parse_location(resource):
    address = resource.get("address", {})

    return {
        "resourceType": "Location",
        "id": resource.get("id"),
        "name": resource.get("name"),
        "type": _extract_codeable_text(resource.get("type", [{}])[0]) if resource.get("type") else None,
        "address": address.get("text") or ", ".join(filter(None, address.get("line", []))),
        "managingOrganization": resource.get("managingOrganization", {}).get("display"),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _first_coding_display(codeable_concept):
    for c in codeable_concept.get("coding", []):
        if c.get("display"):
            return c["display"]
    return None


def _extract_codeable_text(cc):
    if not cc or not isinstance(cc, dict):
        return None
    return cc.get("text") or _first_coding_display(cc)


def _extract_location_ref(encounter):
    for loc in encounter.get("location", []):
        ref = loc.get("location", {}).get("reference")
        if ref:
            return ref
    return None


def _extract_obs_category(resource):
    for cat in resource.get("category", []):
        for c in cat.get("coding", []):
            if c.get("display"):
                return c["display"]
            if c.get("code"):
                return c["code"]
    return None


def _extract_dr_category(resource):
    for cat in resource.get("category", []):
        text = cat.get("text") or _first_coding_display(cat)
        if text:
            return text
    return None


# ---------------------------------------------------------------------------
# Clinical Unit Detection (legacy single Condition mode)
# ---------------------------------------------------------------------------

UNIT_KEYWORDS = {
    "Mental Health": [
        "depression", "anxiety", "suicidal", "mental", "psychiatric",
        "bipolar", "schizophrenia", "self-harm", "mood disorder",
    ],
    "ICU": [
        "respiratory failure", "sepsis", "shock", "multi-organ", "ventilator",
        "critical illness", "hemodynamic", "intensive care", "acute respiratory distress",
    ],
    "Cardiology": [
        "heart failure", "myocardial infarction", "arrhythmia", "atrial fibrillation",
        "angina", "coronary", "cardiac", "chest pain", "hypertension",
    ],
    "Emergency": [
        "trauma", "emergency", "acute pain", "fall", "injury", "laceration",
        "fracture", "ed visit", "urgent", "syncope",
    ],
    "Pulmonology": [
        "copd", "asthma", "pneumonia", "bronchitis", "pulmonary", "hypoxia",
    ],
    "Neurology": [
        "stroke", "seizure", "epilepsy", "migraine", "neurological", "tbi",
    ],
    "Oncology": [
        "cancer", "malignant", "tumor", "neoplasm", "oncology", "metastatic",
    ],
}


def detect_clinical_unit(parsed_condition):
    text_parts = []
    if parsed_condition.get("display"):
        text_parts.append(parsed_condition["display"].lower())
    for cat in parsed_condition.get("categories", []):
        if cat:
            text_parts.append(cat.lower())
    for code_system in ["snomed", "icd9", "icd10"]:
        code_data = parsed_condition.get(code_system)
        if code_data and code_data.get("display"):
            text_parts.append(code_data["display"].lower())

    combined = " ".join(text_parts)
    for unit, keywords in UNIT_KEYWORDS.items():
        if any(kw in combined for kw in keywords):
            return unit
    return "General Medicine"


# ---------------------------------------------------------------------------
# Unified Resource Parser (legacy single-resource mode)
# ---------------------------------------------------------------------------

PARSERS = {
    "Condition": parse_condition,
    "Patient": parse_patient,
    "Encounter": parse_encounter,
    "Observation": parse_observation,
    "MedicationRequest": parse_medication_request,
    "Procedure": parse_procedure,
    "DiagnosticReport": parse_diagnostic_report,
    "AllergyIntolerance": parse_allergy_intolerance,
    "Location": parse_location,
}


def parse_resource(resource):
    resource_type = resource.get("resourceType")
    parser = PARSERS.get(resource_type)

    if parser:
        parsed = parser(resource)
        if resource_type == "Condition":
            parsed["clinical_unit"] = detect_clinical_unit(parsed)
        return parsed

    return {
        "resourceType": resource_type,
        "id": resource.get("id"),
    }


# ---------------------------------------------------------------------------
# Request Parsing / Validation
# ---------------------------------------------------------------------------

def _response(status_code, body):
    return {
        "statusCode": status_code,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(body),
    }


def _check_api_key(event):
    if not API_KEY:
        return None
    headers = event.get("headers") or {}
    key = headers.get("x-api-key") or headers.get("X-API-Key") or ""
    if key != API_KEY:
        log.warning("Rejected request — invalid or missing API key")
        return _response(401, {"error": "Unauthorized"})
    return None


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


def _is_phase0_ai_request(body: dict[str, Any]) -> bool:
    if not isinstance(body, dict):
        return False

    if body.get("schemaVersion") != PHASE0_AI_SCHEMA:
        return False

    fhir_aggregate = body.get("fhirAggregate")
    if not isinstance(fhir_aggregate, dict):
        return False

    return fhir_aggregate.get("schemaVersion") == PHASE0_FHIR_SCHEMA


def _validate_phase0_ai_request(body: dict[str, Any]) -> list[str]:
    errors = []

    missing_top = REQUIRED_TOP_LEVEL_KEYS - set(body.keys())
    if missing_top:
        errors.append(f"Missing top-level keys: {sorted(missing_top)}")
        return errors

    if body.get("schemaVersion") != PHASE0_AI_SCHEMA:
        errors.append(f"schemaVersion must be '{PHASE0_AI_SCHEMA}'")

    fhir_aggregate = body.get("fhirAggregate")
    if not isinstance(fhir_aggregate, dict):
        errors.append("fhirAggregate must be an object")
        return errors

    if fhir_aggregate.get("schemaVersion") != PHASE0_FHIR_SCHEMA:
        errors.append(f"fhirAggregate.schemaVersion must be '{PHASE0_FHIR_SCHEMA}'")

    run_context = fhir_aggregate.get("runContext")
    if not isinstance(run_context, dict):
        errors.append("fhirAggregate.runContext must be an object")
    else:
        if body.get("runId") != run_context.get("runId"):
            errors.append("runId must equal fhirAggregate.runContext.runId")

    resource_summary = fhir_aggregate.get("resourceSummary")
    if not isinstance(resource_summary, dict):
        errors.append("fhirAggregate.resourceSummary must be an object")

    resources = fhir_aggregate.get("resources")
    if not isinstance(resources, dict):
        errors.append("fhirAggregate.resources must be an object")
        return errors

    for family in REQUIRED_RESOURCE_FAMILIES:
        if family not in resources:
            errors.append(f"Missing resources.{family}")

    if resource_summary:
        for family in REQUIRED_RESOURCE_FAMILIES:
            if family not in resource_summary:
                errors.append(f"Missing resourceSummary.{family}")

    patient_obj = resources.get("patient")
    if patient_obj is not None and not isinstance(patient_obj, dict):
        errors.append("resources.patient must be an object or null")

    if resource_summary:
        patient_count = resource_summary.get("patient")
        if patient_count not in (0, 1):
            errors.append("resourceSummary.patient must be 0 or 1")
        else:
            expected_patient_count = 0 if patient_obj is None else 1
            if patient_count != expected_patient_count:
                errors.append("resourceSummary.patient does not match resources.patient")

    for family in REQUIRED_RESOURCE_FAMILIES:
        if family == "patient":
            continue

        value = resources.get(family)
        if not isinstance(value, list):
            errors.append(f"resources.{family} must be a list")
            continue

        if resource_summary:
            expected = resource_summary.get(family)
            if isinstance(expected, int) and expected != len(value):
                errors.append(
                    f"resourceSummary.{family}={expected} does not match len(resources.{family})={len(value)}"
                )

    return errors


# ---------------------------------------------------------------------------
# Lambda handler
# ---------------------------------------------------------------------------

def lambda_handler(event, context):
    auth_err = _check_api_key(event)
    if auth_err:
        return auth_err

    try:
        body = _extract_body(event)

        if _is_phase0_ai_request(body):
            validation_errors = _validate_phase0_ai_request(body)
            if validation_errors:
                log.warning("Phase 0 request validation failed: %s", validation_errors)
                return _response(400, {
                    "error": "Invalid phase0.ai.request.v1 payload",
                    "details": validation_errors,
                })

            log.info("Routing validated phase0.ai.request.v1 payload to baseline_analyze_handler")
            from baseline_analyze_handler import lambda_handler as analyze_handler
            return analyze_handler(event, context)

        if body.get("resourceType"):
            parsed = parse_resource(body)
            return _response(200, {
                "message": "FHIR data processed successfully",
                "data": [parsed],
            })

        return _response(400, {"error": "Invalid input"})

    except json.JSONDecodeError:
        log.warning("Received non-JSON body")
        return _response(400, {"error": "Request body must be valid JSON"})
    except ValueError as exc:
        log.warning("Invalid request format: %s", exc)
        return _response(400, {"error": str(exc)})
    except Exception:
        log.exception("Unhandled error in lambda_handler")
        return _response(500, {"error": "Internal server error"})
