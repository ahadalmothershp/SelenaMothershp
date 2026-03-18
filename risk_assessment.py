import boto3
import json
import re
import time
import logging
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from lambda_function import lambda_handler

# -----------------------------
# Config
# -----------------------------
FHIR_ROOT = Path("selena-fhir-test-data")
RESULTS_ROOT = Path("selena-results")
RESULTS_ROOT.mkdir(exist_ok=True)

LOG_GROUP = '/selena/phase0/baseline-runs'
LOG_STREAM = f'baseline-local-{int(time.time())}'

RESOURCE_FOLDERS = {
    "patients": "Patient",
    "encounter": "Encounter",
    "observation": "Observation",
    "condition": "Condition",
    "location": "Location",
    "medicationRequest": "MedicationRequest",
    "procedure": "Procedure",
    "diagnosticReport": "DiagnosticReport",
    "questionnaireResponse": "QuestionnaireResponse",
    "allergyIntolerance": "AllergyIntolerance"
}

# -----------------------------
# CloudWatch Logs Client
# -----------------------------
logs = boto3.client('logs', region_name='us-west-2')
bedrock = boto3.client("bedrock-runtime", region_name="us-west-2")

def log_message(message):
    """Send log to CloudWatch Logs"""
    try:
        # Ensure log stream exists
        logs.create_log_stream(
            logGroupName=LOG_GROUP, 
            logStreamName=LOG_STREAM
        )
    except logs.exceptions.ResourceAlreadyExistsException:
        pass
    except Exception as e:
        print(f"Log stream error: {e}")
        return
    
    try:
        logs.put_log_events(
            logGroupName=LOG_GROUP,
            logStreamName=LOG_STREAM,
            logEvents=[{
                'timestamp': int(time.time() * 1000),
                'message': json.dumps({
                    'timestamp': datetime.utcnow().isoformat() + 'Z',
                    'level': 'INFO',
                    'message': message,
                    'log_stream': LOG_STREAM
                })
            }]
        )
        print(f"LOG: {message}")  # Also print to console
    except Exception as e:
        print(f"Log put error: {e}")

# -----------------------------
# Helpers
# -----------------------------
def load_json_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def extract_entries(data):
    """Supports: FHIR Bundle with entry[], Raw list of resources, Single resource object"""
    if isinstance(data, dict):
        if data.get("resourceType") == "Bundle":
            return [entry["resource"] for entry in data.get("entry", []) if "resource" in entry]
        elif "resourceType" in data:
            return [data]
        elif isinstance(data.get("entry"), list):
            return [entry["resource"] for entry in data["entry"] if "resource" in entry]
    elif isinstance(data, list):
        return data
    return []

def get_reference_id(ref):
    """Extract ID from FHIR reference strings like "Patient/123" or "Encounter/abc" """
    if not ref or not isinstance(ref, str):
        return None
    return ref.split("/")[-1]

def get_patient_id(resource):
    """Extract patient/subject ID from common FHIR fields."""
    if resource.get("resourceType") == "Patient":
        return resource.get("id")
    
    subject = resource.get("subject", {})
    if isinstance(subject, dict):
        ref = subject.get("reference")
        if ref:
            return get_reference_id(ref)
    
    patient = resource.get("patient", {})
    if isinstance(patient, dict):
        ref = patient.get("reference")
        if ref:
            return get_reference_id(ref)
    
    return None

def get_encounter_id(resource):
    encounter = resource.get("encounter", {})
    if isinstance(encounter, dict):
        ref = encounter.get("reference")
        if ref:
            return get_reference_id(ref)
    return None

def run_local_lambda(event):
    """Calls lambda_handler locally. Converts 'body' string JSON to dict automatically."""
    result = lambda_handler(event, None)
    body = result.get("body")
    
    if isinstance(body, str):
        try:
            return json.loads(body)
        except json.JSONDecodeError:
            return {"error": "Lambda returned invalid JSON", "raw_body": body}
    return result

def build_prompt(clinical_data):
    return f"""
You are a healthcare risk assessment assistant.

Analyze the following clinical data and return JSON only.

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

def parse_claude_output(raw_text):
    cleaned = re.sub(r"^```json\s*|\s*```$", "", raw_text.strip(), flags=re.MULTILINE)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return {"error": "Claude did not return valid JSON", "raw_output": raw_text}

def get_claude_risk(clinical_data, patient_id):
    """Get risk assessment from Claude with logging"""
    log_message(f"Calling Claude for patient {patient_id} with {len(clinical_data)} clinical items")
    
    prompt = build_prompt(clinical_data)
    
    try:
        response = bedrock.converse(
            modelId="us.anthropic.claude-sonnet-4-6",
            messages=[{
                "role": "user",
                "content": [{"text": prompt}]
            }],
            inferenceConfig={"maxTokens": 6400, "temperature": 0.1}
        )
        
        text_output = response["output"]["message"]["content"][0]["text"]
        result = parse_claude_output(text_output)
        log_message(f"Claude response for {patient_id}: {result.get('risk_level', 'N/A')}")
        return result
        
    except Exception as e:
        error_msg = f"Claude error for {patient_id}: {str(e)}"
        log_message(error_msg)
        return {"error": error_msg}

# -----------------------------
# Load all resource files
# -----------------------------
def load_all_resources():
    """Loads all known resource files from folder structure"""
    log_message("Starting resource loading")
    resource_map = defaultdict(list)
    
    for folder_name, resource_type in RESOURCE_FOLDERS.items():
        folder_path = FHIR_ROOT / folder_name
        file_path = folder_path / f"{folder_name}.json"
        
        if not file_path.exists():
            log_message(f"WARNING: Missing file {file_path}")
            continue
        
        try:
            raw_data = load_json_file(file_path)
            resources = extract_entries(raw_data)
            resource_map[resource_type].extend(resources)
            log_message(f"Loaded {len(resources)} {resource_type} resources from {file_path}")
        except Exception as e:
            log_message(f"ERROR loading {file_path}: {e}")
    
    log_message(f"Resource loading complete. Total types: {len(resource_map)}")
    return resource_map

# -----------------------------
# Build per-patient bundles
# -----------------------------
def build_patient_bundles(resource_map):
    """Create one FHIR Bundle per patient by linking all resources by patient ID."""
    log_message("Building patient bundles")
    
    patients = resource_map.get("Patient", [])
    encounters = resource_map.get("Encounter", [])
    observations = resource_map.get("Observation", [])
    conditions = resource_map.get("Condition", [])
    locations = resource_map.get("Location", [])
    medications = resource_map.get("MedicationRequest", [])
    procedures = resource_map.get("Procedure", [])
    diagnostic_reports = resource_map.get("DiagnosticReport", [])
    questionnaires = resource_map.get("QuestionnaireResponse", [])
    allergies = resource_map.get("AllergyIntolerance", [])
    
    # Index encounters by patient
    encounters_by_patient = defaultdict(list)
    for enc in encounters:
        pid = get_patient_id(enc)
        if pid:
            encounters_by_patient[pid].append(enc)
    
    # Index other resources by patient
    def index_by_patient(resources):
        idx = defaultdict(list)
        for r in resources:
            pid = get_patient_id(r)
            if pid:
                idx[pid].append(r)
        return idx
    
    observations_by_patient = index_by_patient(observations)
    conditions_by_patient = index_by_patient(conditions)
    medications_by_patient = index_by_patient(medications)
    procedures_by_patient = index_by_patient(procedures)
    diagnostic_reports_by_patient = index_by_patient(diagnostic_reports)
    questionnaires_by_patient = index_by_patient(questionnaires)
    allergies_by_patient = index_by_patient(allergies)
    
    # Locations via encounter.location.reference
    location_lookup = {loc.get("id"): loc for loc in locations if loc.get("id")}
    locations_by_patient = defaultdict(list)
    
    for pid, patient_encounters in encounters_by_patient.items():
        seen = set()
        for enc in patient_encounters:
            for loc_ref in enc.get("location", []):
                if isinstance(loc_ref, dict):
                    loc_obj = loc_ref.get("location", {})
                    if isinstance(loc_obj, dict):
                        loc_id = get_reference_id(loc_obj.get("reference"))
                        if loc_id and loc_id in location_lookup and loc_id not in seen:
                            locations_by_patient[pid].append(location_lookup[loc_id])
                            seen.add(loc_id)
    
    # Build one bundle per patient
    bundles = []
    for patient in patients:
        pid = patient.get("id")
        if not pid:
            continue
        
        patient_resources = [patient]
        patient_resources.extend(encounters_by_patient.get(pid, []))
        patient_resources.extend(locations_by_patient.get(pid, []))
        patient_resources.extend(observations_by_patient.get(pid, []))
        patient_resources.extend(conditions_by_patient.get(pid, []))
        patient_resources.extend(medications_by_patient.get(pid, []))
        patient_resources.extend(procedures_by_patient.get(pid, []))
        patient_resources.extend(diagnostic_reports_by_patient.get(pid, []))
        patient_resources.extend(questionnaires_by_patient.get(pid, []))
        patient_resources.extend(allergies_by_patient.get(pid, []))
        
        bundle = {
            "resourceType": "Bundle",
            "type": "collection",
            "id": f"patient-bundle-{pid}",
            "entry": [{"resource": r} for r in patient_resources]
        }
        
        bundles.append({
            "patient_id": pid,
            "patient_name": extract_patient_name(patient),
            "bundle": bundle,
            "resource_counts": count_bundle_resources(patient_resources)
        })
    
    log_message(f"Built {len(bundles)} patient bundles")
    return bundles

def extract_patient_name(patient):
    names = patient.get("name", [])
    if names and isinstance(names, list):
        n = names[0]
        given = " ".join(n.get("given", [])) if isinstance(n.get("given"), list) else ""
        family = n.get("family", "")
        full = f"{given} {family}".strip()
        return full if full else patient.get("id", "Unknown")
    return patient.get("id", "Unknown")

def count_bundle_resources(resources):
    counts = defaultdict(int)
    for r in resources:
        rt = r.get("resourceType", "Unknown")
        counts[rt] += 1
    return dict(counts)

# -----------------------------
# Process each patient bundle
# -----------------------------
def process_patient_bundle(patient_bundle_info):
    patient_id = patient_bundle_info["patient_id"]
    patient_name = patient_bundle_info["patient_name"]
    bundle = patient_bundle_info["bundle"]
    
    log_message(f"Processing patient {patient_id} ({patient_name}): {len(bundle['entry'])} resources")
    
    try:
        parsed_output = run_local_lambda(bundle)
        clinical_data = parsed_output.get("data", [])
        
        log_message(f"Lambda extracted {len(clinical_data)} clinical items for {patient_id}")
        
        if not clinical_data:
            risk_result = {
                "risk_level": "normal",
                "summary": "No clinical data extracted by lambda.",
                "key_findings": [],
                "risk_factors": [],
                "recommendation": "Review parser output; insufficient structured data.",
                "data_sufficiency": "limited"
            }
            log_message(f"No clinical data for {patient_id}, using default risk")
        else:
            risk_result = get_claude_risk(clinical_data, patient_id)
        
        return {
            "patient_id": patient_id,
            "patient_name": patient_name,
            "processed_at": datetime.utcnow().isoformat() + "Z",
            "resource_counts": patient_bundle_info["resource_counts"],
            "bundle_id": bundle.get("id"),
            "lambda_output": parsed_output,
            "claude_risk": risk_result
        }
    
    except Exception as e:
        error_msg = f"Error processing {patient_id}: {str(e)}"
        log_message(error_msg)
        return {
            "patient_id": patient_id,
            "patient_name": patient_name,
            "processed_at": datetime.utcnow().isoformat() + "Z",
            "resource_counts": patient_bundle_info["resource_counts"],
            "bundle_id": bundle.get("id"),
            "error": str(e)
        }

def save_patient_result(result):
    out_file = RESULTS_ROOT / f"{result['patient_id']}-result.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    log_message(f"Saved result for {result['patient_id']} to {out_file}")
    return out_file

# -----------------------------
# Main
# -----------------------------
def main():
    log_message("=== Starting FHIR baseline batch processing ===")
    log_message(f"Using log group: {LOG_GROUP}, stream: {LOG_STREAM}")
    
    # Load resources
    resource_map = load_all_resources()
    
    # Build patient bundles
    patient_bundles = build_patient_bundles(resource_map)
    
    if not patient_bundles:
        log_message("ERROR: No patient bundles could be built. Check patient IDs and references.")
        return
    
    log_message(f"Starting processing of {len(patient_bundles)} patient bundles")
    
    summary = {
        "processed_at": datetime.utcnow().isoformat() + "Z",
        "total_patients": len(patient_bundles),
        "processed": 0,
        "success": 0,
        "failed": 0,
        "patients": [],
        "log_group": LOG_GROUP,
        "log_stream": LOG_STREAM
    }
    
    for idx, pb in enumerate(patient_bundles, start=1):
        log_message(f"[BATCH {idx}/{len(patient_bundles)}] Processing patient {pb['patient_id']} ({pb['patient_name']})")
        
        result = process_patient_bundle(pb)
        output_file = save_patient_result(result)
        
        summary["processed"] += 1
        
        if "error" in result:
            summary["failed"] += 1
            status = "FAILED"
        else:
            summary["success"] += 1
            status = "OK"
        
        summary["patients"].append({
            "patient_id": result["patient_id"],
            "patient_name": result["patient_name"],
            "output": str(output_file),
            "status": status,
            "risk_level": result.get("claude_risk", {}).get("risk_level"),
            "resource_counts": result.get("resource_counts", {})
        })
        
        log_message(f"  -> {status}: {output_file} (risk: {result.get('claude_risk', {}).get('risk_level', 'N/A')})")
    
    # Save summary
    summary_file = RESULTS_ROOT / "batch-summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    
    log_message("=== Batch processing COMPLETE ===")
    log_message(json.dumps({
        "total_patients": summary["total_patients"],
        "processed": summary["processed"],
        "success": summary["success"],
        "failed": summary["failed"],
        "summary_file": str(summary_file),
        "log_group": LOG_GROUP,
        "log_stream": LOG_STREAM
    }, indent=2))

if __name__ == "__main__":
    main()
