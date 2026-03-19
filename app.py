import json
import os
import logging
from functools import wraps

from dotenv import load_dotenv
from flask import Flask, request, jsonify, g

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
log = logging.getLogger("selena")

API_KEY = os.getenv("API_KEY")
ALLOWED_ORIGIN = os.getenv("ALLOWED_ORIGIN", "*")
MAX_CONTENT_LENGTH = int(os.getenv("MAX_CONTENT_LENGTH", 1_048_576))

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH


# ---------------------------------------------------------------------------
# Auth middleware
# ---------------------------------------------------------------------------
def require_api_key(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not API_KEY:
            return f(*args, **kwargs)

        key = request.headers.get("X-API-Key") or request.headers.get("Authorization", "").removeprefix("Bearer ")
        if key != API_KEY:
            log.warning("Rejected request — invalid or missing API key from %s", request.remote_addr)
            return jsonify({"error": "Unauthorized"}), 401
        return f(*args, **kwargs)
    return decorated


# ---------------------------------------------------------------------------
# CORS — applied to every response
# ---------------------------------------------------------------------------
@app.after_request
def add_cors_and_security_headers(resp):
    resp.headers["Access-Control-Allow-Origin"] = ALLOWED_ORIGIN
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type, X-API-Key, Authorization"
    resp.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
    resp.headers["X-Content-Type-Options"] = "nosniff"
    resp.headers["X-Frame-Options"] = "DENY"
    return resp


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/baseline/analyze", methods=["POST", "OPTIONS"])
@require_api_key
def baseline_analyze():
    if request.method == "OPTIONS":
        return "", 204

    body = request.get_json(silent=True)
    if body is None:
        log.warning("Bad request — body is not valid JSON")
        return jsonify({"error": "Request body must be valid JSON"}), 400

    if not body.get("schemaVersion", "").startswith("phase0"):
        log.warning("Bad request — missing or unrecognised schemaVersion")
        return jsonify({"error": "Missing or unrecognised schemaVersion"}), 400

    if not body.get("fhirAggregate"):
        log.warning("Bad request — missing fhirAggregate")
        return jsonify({"error": "Missing fhirAggregate in request body"}), 400

    from lambda_function import lambda_handler

    event = {
        "body": json.dumps(body),
        "httpMethod": "POST",
        "path": "/baseline/analyze",
    }

    log.info(
        "Processing /baseline/analyze — runId=%s patientId=%s",
        body.get("runId"),
        body.get("fhirAggregate", {}).get("runContext", {}).get("patientId"),
    )

    result = lambda_handler(event, None)
    resp_body = json.loads(result["body"]) if isinstance(result["body"], str) else result["body"]
    return jsonify(resp_body), result["statusCode"]


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


# ---------------------------------------------------------------------------
# Error handlers
# ---------------------------------------------------------------------------
@app.errorhandler(413)
def payload_too_large(e):
    log.warning("Payload too large from %s", request.remote_addr)
    return jsonify({"error": f"Payload exceeds {MAX_CONTENT_LENGTH} byte limit"}), 413


@app.errorhandler(500)
def internal_error(e):
    log.exception("Unhandled server error")
    return jsonify({"error": "Internal server error"}), 500


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    log.info("Starting Selena dev server on :8000")
    app.run(host="0.0.0.0", port=8000)
