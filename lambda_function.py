import json

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

    categories = [cat.get("text") for cat in condition.get("category", [])]

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
        "recordedDate": condition.get("recordedDate")
    }

def parse_resource(resource):
    resource_type = resource.get("resourceType")

    if resource_type == "Condition":
        return parse_condition(resource)

    return {
        "resourceType": resource_type,
        "id": resource.get("id"),
        "raw": resource
    }

def response(status_code, body):
    return {
        "statusCode": status_code,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(body, indent=2)
    }

def lambda_handler(event, context):
    try:
        if event.get("resourceType"):
            parsed = parse_resource(event)
            return response(200, {
                "message": "FHIR data processed successfully",
                "count": 1,
                "data": [parsed]
            })

        return response(400, {
            "error": "Invalid input"
        })

    except Exception as e:
        return response(500, {
            "error": str(e)
        })
