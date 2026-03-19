import json
import logging

log = logging.getLogger("selena.core")


def parse_condition(condition):
    code = condition.get("code", {})
    coding = code.get("coding", [])

    snomed = None
    icd9 = None
    icd10 = None

    for c in coding:
        system = c.get("system")
        if system == "http://snomed.info/sct":
            snomed = {"code": c.get("code"), "display": c.get("display")}
        elif system == "http://hl7.org/fhir/sid/icd-9-cm":
            icd9 = {"code": c.get("code"), "display": c.get("display")}
        elif system == "http://hl7.org/fhir/sid/icd-10-cm":
            icd10 = {"code": c.get("code"), "display": c.get("display")}

    categories = [cat.get("text") for cat in condition.get("category", []) if cat.get("text")]

    return {
        "resourceType": "Condition",
        "id": condition.get("id"),
        "display": code.get("text"),
        "categories": categories,
        "snomed": snomed,
        "icd9": icd9,
        "icd10": icd10,
        "subject_reference": condition.get("subject", {}).get("reference"),
        "subject_display": condition.get("subject", {}).get("display"),
        "encounter_display": condition.get("encounter", {}).get("display"),
        "encounter_identifier": condition.get("encounter", {}).get("identifier", {}).get("value"),
        "recordedDate": condition.get("recordedDate"),
    }


def detect_clinical_unit(parsed_resource):
    """
    Determine which clinical unit the patient likely falls into
    based on condition display, coding display, and category text.
    """
    text_parts = []

    if parsed_resource.get("display"):
        text_parts.append(parsed_resource["display"].lower())

    for cat in parsed_resource.get("categories", []):
        if cat:
            text_parts.append(cat.lower())

    for code_system in ["snomed", "icd9", "icd10"]:
        code_data = parsed_resource.get(code_system)
        if code_data and code_data.get("display"):
            text_parts.append(code_data["display"].lower())

    combined_text = " ".join(text_parts)

    unit_keywords = {
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

    for unit, keywords in unit_keywords.items():
        if any(kw in combined_text for kw in keywords):
            return unit

    return "Unknown / General Medicine"


def parse_resource(resource):
    resource_type = resource.get("resourceType")

    if resource_type == "Condition":
        parsed = parse_condition(resource)
        parsed["clinical_unit"] = detect_clinical_unit(parsed)
        return parsed

    return {
        "resourceType": resource_type,
        "id": resource.get("id"),
        "clinical_unit": "Unknown / Unsupported Resource",
    }


def _response(status_code, body):
    return {
        "statusCode": status_code,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(body),
    }


def lambda_handler(event, context):
    try:
        body = event
        if isinstance(event.get("body"), str):
            body = json.loads(event["body"])

        if body.get("schemaVersion", "").startswith("phase0.syed") or body.get("fhirAggregate"):
            from baseline_analyze_handler import lambda_handler as analyze_handler
            return analyze_handler(event, context)

        if body.get("resourceType"):
            parsed = parse_resource(body)
            return _response(200, {
                "message": "FHIR data processed successfully",
                "count": 1,
                "data": [parsed],
                "clinical_unit": parsed.get("clinical_unit"),
            })

        return _response(400, {"error": "Invalid input"})

    except json.JSONDecodeError:
        log.warning("Received non-JSON body")
        return _response(400, {"error": "Request body must be valid JSON"})
    except Exception:
        log.exception("Unhandled error in lambda_handler")
        return _response(500, {"error": "Internal server error"})
