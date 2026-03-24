import json
import logging
import os

log = logging.getLogger("selena.core")

API_KEY = os.environ.get("API_KEY")


# ---------------------------------------------------------------------------
# FHIR Resource Parsers
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
    position = resource.get("position", {})

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
# Clinical Unit Detection (Condition-specific)
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
# Unified Resource Parser
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
# Lambda handler
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


def lambda_handler(event, context):
    auth_err = _check_api_key(event)
    if auth_err:
        return auth_err

    try:
        body = event
        if isinstance(event.get("body"), str):
            body = json.loads(event["body"])

        if body.get("schemaVersion", "").startswith("phase0.") or body.get("fhirAggregate"):
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
    except Exception:
        log.exception("Unhandled error in lambda_handler")
        return _response(500, {"error": "Internal server error"})
