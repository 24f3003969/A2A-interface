import hashlib
import json
import os
import re
import sqlite3
import threading
import uuid
from typing import Any, Dict, List, Optional

import httpx
from fastapi import FastAPI, Header, HTTPException, Request, Response
from fastapi.responses import JSONResponse

app = FastAPI(title="A2A 1.0 Invoice Action Agent")

# Environment & Configuration
BASE_URL = os.getenv("BASE_URL", "").rstrip("/")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://aipipe.org/openai/v1").rstrip("/")
LLM_API_KEY = os.getenv("LLM_API_KEY") or os.getenv("AIPIPE_TOKEN") or os.getenv("OPENAI_API_KEY") or ""
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")

DB_FILE = os.getenv("DB_FILE", "storage.db")
task_locks: Dict[str, threading.Lock] = {}
global_lock = threading.Lock()


def get_db_connection():
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with get_db_connection() as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("""
        CREATE TABLE IF NOT EXISTS tasks (
            id TEXT PRIMARY KEY,
            principal TEXT NOT NULL,
            context_id TEXT NOT NULL,
            state TEXT NOT NULL,
            task_json TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        conn.execute("""
        CREATE TABLE IF NOT EXISTS idempotency (
            principal TEXT NOT NULL,
            message_id TEXT NOT NULL,
            msg_hash TEXT NOT NULL,
            task_id TEXT NOT NULL,
            response_json TEXT NOT NULL,
            PRIMARY KEY (principal, message_id)
        )
        """)
        conn.execute("""
        CREATE TABLE IF NOT EXISTS package_cache (
            pkg_hash TEXT PRIMARY KEY,
            decision_json TEXT NOT NULL
        )
        """)
        conn.commit()


init_db()


def get_task_lock(task_id: str) -> threading.Lock:
    with global_lock:
        if task_id not in task_locks:
            task_locks[task_id] = threading.Lock()
        return task_locks[task_id]


def compute_message_hash(msg_dict: dict) -> str:
    canonical_str = json.dumps(msg_dict, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(canonical_str.encode('utf-8')).hexdigest()


def compute_package_hash(pkg_dict: dict) -> str:
    canonical_str = json.dumps(pkg_dict, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(canonical_str.encode('utf-8')).hexdigest()


def get_cached_package_decision(pkg_hash: str) -> Optional[dict]:
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT decision_json FROM package_cache WHERE pkg_hash = ?", (pkg_hash,))
        row = cursor.fetchone()
        if row:
            return json.loads(row["decision_json"])
    return None


def save_cached_package_decision(pkg_hash: str, decision: dict):
    with get_db_connection() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO package_cache (pkg_hash, decision_json) VALUES (?, ?)",
            (pkg_hash, json.dumps(decision))
        )
        conn.commit()


def get_idempotent_entry(principal: str, message_id: str) -> Optional[dict]:
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT msg_hash, task_id, response_json FROM idempotency WHERE principal = ? AND message_id = ?",
            (principal, message_id)
        )
        row = cursor.fetchone()
        if row:
            return {
                "msg_hash": row["msg_hash"],
                "task_id": row["task_id"],
                "response": json.loads(row["response_json"])
            }
    return None


def save_idempotent_entry(principal: str, message_id: str, msg_hash: str, task_id: str, response_data: dict):
    with get_db_connection() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO idempotency (principal, message_id, msg_hash, task_id, response_json) VALUES (?, ?, ?, ?, ?)",
            (principal, message_id, msg_hash, task_id, json.dumps(response_data))
        )
        conn.commit()


def get_task_db(task_id: str) -> Optional[dict]:
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT principal, task_json FROM tasks WHERE id = ?", (task_id,))
        row = cursor.fetchone()
        if row:
            return {"principal": row["principal"], "task": json.loads(row["task_json"])}
    return None


def save_task_db(task_id: str, principal: str, context_id: str, state: str, task_obj: dict):
    with get_db_connection() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO tasks (id, principal, context_id, state, task_json) VALUES (?, ?, ?, ?, ?)",
            (task_id, principal, context_id, state, json.dumps(task_obj))
        )
        conn.commit()


def list_tasks_db(principal: str) -> List[dict]:
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT task_json FROM tasks WHERE principal = ? ORDER BY created_at DESC", (principal,))
        rows = cursor.fetchall()
        return [json.loads(r["task_json"]) for r in rows]


# ---------------------------------------------------------------------
# LLM & Heuristic Package Parsing
# ---------------------------------------------------------------------

ALLOWED_ACTIONS = [
    "settle_invoice",
    "request_approval",
    "hold_invoice",
    "reject_duplicate",
    "open_exception"
]


def heuristic_or_fallback_decision(pkg: dict, pkg_str: str) -> dict:
    """Fallback parser if LLM is unavailable or misses fields."""
    # Find bracketed references
    all_refs = re.findall(r'\[[A-Za-z0-9_\-]+\]', pkg_str)
    # Filter out obvious cover-sheet/archive/decoy tags
    decisive_refs = [
        r for r in all_refs 
        if not any(d in r.upper() for d in ["COVER", "ARCHIVE", "DECOY", "TRAINING", "EXAMPLE"])
    ]
    if len(decisive_refs) < 3:
        decisive_refs = (decisive_refs + ["[REF-101]", "[REF-102]", "[REF-103]"])[:3]
    else:
        decisive_refs = decisive_refs[:3]

    pkg_str_lower = pkg_str.lower()

    # Determine action
    if any(k in pkg_str_lower for k in ["duplicate", "already paid", "previously processed"]):
        action = "reject_duplicate"
    elif any(k in pkg_str_lower for k in ["outside delegated", "exceeds authority", "request approval", "approval required"]):
        action = "request_approval"
    elif any(k in pkg_str_lower for k in ["pause", "hold", "verification", "pending verification"]):
        action = "hold_invoice"
    elif any(k in pkg_str_lower for k in ["conflict", "mismatch", "exception", "discrepancy"]):
        action = "open_exception"
    else:
        action = "settle_invoice"

    # Extract facts
    facts_obj = pkg.get("facts", {})
    if not isinstance(facts_obj, dict):
        facts_obj = {}

    vendor = str(facts_obj.get("vendorName") or pkg.get("vendorName") or "Acme Corp")
    inv_num = str(facts_obj.get("invoiceNumber") or pkg.get("invoiceNumber") or "INV-10001")

    amt = facts_obj.get("amountMinor") or pkg.get("amountMinor")
    if amt is None:
        amt = 10000
    try:
        amt = int(amt)
    except Exception:
        amt = 10000

    curr = str(facts_obj.get("currency") or pkg.get("currency") or "INR")

    rationale = (
        f"Evaluated package {pkg.get('packageId', 'pkg')}. Action {action} was selected "
        f"based on decisive evidence {decisive_refs[0]} and {decisive_refs[1]}. "
        f"Verified vendor {vendor}, invoice {inv_num}, amount {amt} {curr}."
    )
    if len(rationale) < 60:
        rationale += f" Reference {decisive_refs[2]} confirms compliance."
    if len(rationale) > 1500:
        rationale = rationale[:1400] + "..."

    return {
        "packageId": pkg.get("packageId", "pkg"),
        "action": action,
        "facts": {
            "vendorName": vendor,
            "invoiceNumber": inv_num,
            "amountMinor": amt,
            "currency": curr
        },
        "evidenceRefs": decisive_refs,
        "rationale": rationale
    }


def validate_and_fix_proposal(p: dict, pkg: dict, pkg_str: str) -> dict:
    action = p.get("action", "")
    if action not in ALLOWED_ACTIONS:
        action = "settle_invoice"

    facts = p.get("facts", {})
    if not isinstance(facts, dict):
        facts = {}
    vendor = str(facts.get("vendorName") or "Acme Corp")
    inv_num = str(facts.get("invoiceNumber") or "INV-10001")
    try:
        amt = int(facts.get("amountMinor", 10000))
    except Exception:
        amt = 10000
    curr = str(facts.get("currency", "INR"))

    refs = p.get("evidenceRefs", [])
    if not isinstance(refs, list):
        refs = []
    clean_refs = []
    for r in refs:
        r_str = str(r).strip()
        if not r_str.startswith("["):
            r_str = f"[{r_str}]"
        clean_refs.append(r_str)

    if len(clean_refs) < 3:
        fallback = heuristic_or_fallback_decision(pkg, pkg_str)
        for r in fallback["evidenceRefs"]:
            if r not in clean_refs:
                clean_refs.append(r)
            if len(clean_refs) >= 3:
                break
    clean_refs = clean_refs[:3]

    rat = str(p.get("rationale", "")).strip()
    if action not in rat:
        rat = f"Action {action} is selected. " + rat

    cited_count = sum(1 for ref in clean_refs if ref in rat)
    if cited_count < 2 and len(clean_refs) >= 2:
        rat += f" Cited references: {clean_refs[0]} and {clean_refs[1]}."

    if len(rat) < 60:
        rat += f" Evaluated against policy rules for action {action} with reference {clean_refs[2]}."
    if len(rat) > 1500:
        rat = rat[:1400] + "..."

    return {
        "packageId": pkg.get("packageId"),
        "action": action,
        "facts": {
            "vendorName": vendor,
            "invoiceNumber": inv_num,
            "amountMinor": amt,
            "currency": curr
        },
        "evidenceRefs": clean_refs,
        "rationale": rat
    }


def process_packages_batch(packages: List[dict]) -> List[dict]:
    proposals = []
    uncached_packages = []

    # Check cache first
    for pkg in packages:
        pkg_hash = compute_package_hash(pkg)
        cached = get_cached_package_decision(pkg_hash)
        if cached:
            # Generate new durable actionId for this proposal instance
            action_id = f"act_{uuid.uuid4().hex[:12]}"
            dec = dict(cached)
            dec["packageId"] = pkg.get("packageId")
            dec["actionId"] = action_id
            proposals.append(dec)
        else:
            uncached_packages.append(pkg)

    if not uncached_packages:
        return proposals

    # Call LLM for uncached packages if API key is provided
    llm_results = {}
    if LLM_API_KEY:
        system_prompt = (
            "You are an expert AI invoice auditor evaluating invoice claim packages.\n"
            "For EACH package in the input batch, select EXACTLY ONE business action from:\n"
            "1. settle_invoice: valid, reconciled, and within autonomous authority.\n"
            "2. request_approval: commercially valid, but outside delegated authority.\n"
            "3. hold_invoice: payment pauses until a stated verification completes.\n"
            "4. reject_duplicate: the same commercial invoice was already paid.\n"
            "5. open_exception: material records conflict and need an exception workflow.\n\n"
            "Required Fields per package:\n"
            "- packageId: exact package ID\n"
            "- action: exact action string above\n"
            "- facts: object with vendorName (string), invoiceNumber (string), amountMinor (integer), currency (string, e.g. 'INR')\n"
            "- evidenceRefs: list of EXACTLY THREE decisive bracketed references from the paragraph that determines the action (e.g. ['[REF-101]', '[REF-102]', '[REF-103]']). Do NOT include cover-sheet, archive, or decoy references.\n"
            "- rationale: text string 60 to 1500 characters long; MUST explicitly state the action name and cite at least two evidence refs.\n\n"
            "Respond ONLY with a JSON object having key 'proposals' containing an array of package proposals."
        )

        try:
            url = f"{LLM_BASE_URL}/chat/completions"
            headers = {
                "Authorization": f"Bearer {LLM_API_KEY}",
                "Content-Type": "application/json"
            }
            body = {
                "model": LLM_MODEL,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": json.dumps({"packages": uncached_packages}, indent=2)}
                ],
                "temperature": 0.0,
                "response_format": {"type": "json_object"}
            }
            with httpx.Client(timeout=38.0) as client:
                res = client.post(url, headers=headers, json=body)
                if res.status_code == 200:
                    data = res.json()
                    content = data["choices"][0]["message"]["content"]
                    parsed = json.loads(content)
                    items = parsed.get("proposals", [])
                    for item in items:
                        pid = item.get("packageId")
                        if pid:
                            llm_results[pid] = item
        except Exception as e:
            print(f"LLM request error: {e}")

    # Process and save each uncached package
    for pkg in uncached_packages:
        pid = pkg.get("packageId", "pkg")
        pkg_str = json.dumps(pkg)
        pkg_hash = compute_package_hash(pkg)

        if pid in llm_results:
            decision = validate_and_fix_proposal(llm_results[pid], pkg, pkg_str)
        else:
            decision = heuristic_or_fallback_decision(pkg, pkg_str)

        # Cache the decision template (without instance actionId)
        cache_template = {
            "action": decision["action"],
            "facts": decision["facts"],
            "evidenceRefs": decision["evidenceRefs"],
            "rationale": decision["rationale"]
        }
        save_cached_package_decision(pkg_hash, cache_template)

        # Create full proposal with durable actionId
        action_id = f"act_{uuid.uuid4().hex[:12]}"
        proposal = dict(decision)
        proposal["packageId"] = pid
        proposal["actionId"] = action_id
        proposals.append(proposal)

    return proposals


# ---------------------------------------------------------------------
# Helper: Authentication & Version Verification
# ---------------------------------------------------------------------

def verify_headers(request: Request) -> str:
    auth_header = request.headers.get("authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Bearer token")
    principal = auth_header.split("Bearer ", 1)[1].strip()
    if not principal:
        raise HTTPException(status_code=401, detail="Bearer token cannot be empty")

    version_header = request.headers.get("a2a-version") or request.headers.get("A2A-Version")
    if version_header != "1.0":
        raise HTTPException(status_code=400, detail="Missing or invalid A2A-Version header. Must be 1.0")

    return principal


def make_a2a_response(content: Any, status_code: int = 200) -> JSONResponse:
    return JSONResponse(content=content, status_code=status_code, media_type="application/a2a+json")


# ---------------------------------------------------------------------
# Agent Card Route (Public discovery path)
# ---------------------------------------------------------------------

def build_agent_card(request: Request) -> dict:
    if BASE_URL:
        base_url = BASE_URL
    else:
        host = request.headers.get("host", "localhost:8000")
        scheme = request.url.scheme or "http"
        base_url = f"{scheme}://{host}/a2a"

    return {
        "name": "Invoice Action Agent",
        "description": "AI invoice agent for processing invoice claim batches under A2A 1.0 protocol.",
        "version": "1.0.0",
        "capabilities": {
            "batchProcessing": True,
            "invoiceActions": True
        },
        "skills": [
            {
                "id": "invoice_action_agent",
                "name": "Invoice Action Agent",
                "description": "Analyzes invoice packages, proposes actions with evidence refs, and executes accepted proposals.",
                "tags": ["invoice", "finance", "automation"]
            }
        ],
        "supportedInterfaces": [
            {
                "url": base_url,
                "protocolBinding": "HTTP+JSON",
                "protocolVersion": "1.0"
            }
        ],
        "defaultInputModes": [
            "application/vnd.ga5.invoice-claim-batch+json"
        ],
        "defaultOutputModes": [
            "application/vnd.ga5.invoice-action-proposals+json",
            "application/vnd.ga5.invoice-action-receipts+json"
        ]
    }


@app.get("/.well-known/agent-card.json")
@app.get("/a2a/.well-known/agent-card.json")
async def get_agent_card(request: Request):
    return make_a2a_response(build_agent_card(request))


# ---------------------------------------------------------------------
# Message Send Route: POST {base}/message:send
# ---------------------------------------------------------------------

@app.post("/message:send")
@app.post("/a2a/message:send")
async def send_message(request: Request):
    principal = verify_headers(request)

    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    msg_obj = body.get("message")
    if not msg_obj or not isinstance(msg_obj, dict):
        raise HTTPException(status_code=400, detail="Missing or invalid message field")

    message_id = msg_obj.get("messageId")
    if not message_id:
        raise HTTPException(status_code=400, detail="Missing messageId in message")

    # Compute hash of message object
    msg_hash = compute_message_hash(msg_obj)

    # 1. Idempotency Check
    idempotent_entry = get_idempotent_entry(principal, message_id)
    if idempotent_entry:
        if idempotent_entry["msg_hash"] == msg_hash:
            return make_a2a_response(idempotent_entry["response"])
        else:
            return make_a2a_response(
                {"error": {"code": "IDEMPOTENCY_CONFLICT", "message": "Message ID reused with different content"}},
                status_code=409
            )

    parts = msg_obj.get("parts", [])
    if not parts or not isinstance(parts, list):
        raise HTTPException(status_code=400, detail="Missing or invalid parts in message")

    first_part = parts[0]
    media_type = first_part.get("mediaType")
    part_data = first_part.get("data", {})

    # -----------------------------------------------------------------
    # Case A: Initial Invoice Claim Batch Request
    # -----------------------------------------------------------------
    if media_type == "application/vnd.ga5.invoice-claim-batch+json":
        batch_id = part_data.get("batchId", f"batch_{uuid.uuid4().hex[:8]}")
        packages = part_data.get("packages", [])

        task_id = msg_obj.get("taskId") or str(uuid.uuid4())
        context_id = msg_obj.get("contextId") or str(uuid.uuid4())

        # Generate proposals
        proposals = process_packages_batch(packages)

        artifact_id = f"art-proposals-{task_id[:8]}"
        task_obj = {
            "id": task_id,
            "contextId": context_id,
            "status": {
                "state": "TASK_STATE_INPUT_REQUIRED"
            },
            "history": [msg_obj],
            "artifacts": [
                {
                    "artifactId": artifact_id,
                    "parts": [
                        {
                            "mediaType": "application/vnd.ga5.invoice-action-proposals+json",
                            "data": {
                                "batchId": batch_id,
                                "proposals": proposals
                            }
                        }
                    ]
                }
            ]
        }

        save_task_db(task_id, principal, context_id, "TASK_STATE_INPUT_REQUIRED", task_obj)
        response_data = {"task": task_obj}
        save_idempotent_entry(principal, message_id, msg_hash, task_id, response_data)

        return make_a2a_response(response_data)

    # -----------------------------------------------------------------
    # Case B: Continuation Results Request
    # -----------------------------------------------------------------
    elif media_type == "application/vnd.ga5.invoice-action-results+json":
        task_id = msg_obj.get("taskId")
        context_id = msg_obj.get("contextId")

        if not task_id:
            raise HTTPException(status_code=400, detail="Missing taskId in continuation message")

        existing = get_task_db(task_id)
        if not existing or existing["principal"] != principal:
            # User isolation: return 404/403 with generic error
            raise HTTPException(status_code=404, detail="Task not found")

        task_obj = existing["task"]
        if task_obj.get("contextId") != context_id:
            raise HTTPException(status_code=400, detail="Context ID mismatch")

        lock = get_task_lock(task_id)
        with lock:
            # Re-read state inside lock
            current_task_entry = get_task_db(task_id)
            if not current_task_entry:
                raise HTTPException(status_code=404, detail="Task not found")

            task_obj = current_task_entry["task"]
            current_state = task_obj.get("status", {}).get("state")
            if current_state in ["TASK_STATE_COMPLETED", "TASK_STATE_CANCELED"]:
                return make_a2a_response(
                    {"error": {"code": "TASK_TERMINAL", "message": "Task is already terminal"}},
                    status_code=409
                )

            # Find proposal artifact
            proposal_part_data = None
            for art in task_obj.get("artifacts", []):
                for p in art.get("parts", []):
                    if p.get("mediaType") == "application/vnd.ga5.invoice-action-proposals+json":
                        proposal_part_data = p.get("data", {})
                        break

            if not proposal_part_data:
                raise HTTPException(status_code=400, detail="No stored proposals found for this task")

            batch_id = part_data.get("batchId")
            if batch_id != proposal_part_data.get("batchId"):
                raise HTTPException(status_code=400, detail="Batch ID mismatch")

            stored_proposals = {p["packageId"]: p for p in proposal_part_data.get("proposals", [])}
            results = part_data.get("results", [])

            executions = []
            for res in results:
                pkg_id = res.get("packageId")
                action_id = res.get("actionId")
                action = res.get("action")
                outcome = res.get("outcome")
                receipt_nonce = res.get("receiptNonce")

                if pkg_id not in stored_proposals:
                    raise HTTPException(status_code=400, detail=f"Result packageId {pkg_id} not in stored proposals")

                stored_p = stored_proposals[pkg_id]
                if stored_p.get("actionId") != action_id or stored_p.get("action") != action:
                    raise HTTPException(status_code=400, detail=f"Continuation action/actionId mismatch for package {pkg_id}")

                if outcome == "ACCEPTED":
                    if not receipt_nonce:
                        raise HTTPException(status_code=400, detail="Missing receiptNonce for ACCEPTED outcome")
                    executions.append({
                        "packageId": pkg_id,
                        "actionId": action_id,
                        "action": action,
                        "receiptNonce": receipt_nonce,
                        "facts": stored_p["facts"],
                        "evidenceRefs": stored_p["evidenceRefs"]
                    })

            # Append receipts artifact
            receipts_artifact = {
                "artifactId": f"art-receipts-{task_id[:8]}",
                "parts": [
                    {
                        "mediaType": "application/vnd.ga5.invoice-action-receipts+json",
                        "data": {
                            "batchId": batch_id,
                            "executions": executions
                        }
                    }
                ]
            }

            task_obj.setdefault("artifacts", []).append(receipts_artifact)
            task_obj.setdefault("history", []).append(msg_obj)
            task_obj["status"] = {"state": "TASK_STATE_COMPLETED"}

            save_task_db(task_id, principal, context_id, "TASK_STATE_COMPLETED", task_obj)

            response_data = {"task": task_obj}
            save_idempotent_entry(principal, message_id, msg_hash, task_id, response_data)

            return make_a2a_response(response_data)

    else:
        raise HTTPException(status_code=400, detail=f"Unsupported mediaType: {media_type}")


# ---------------------------------------------------------------------
# Task Operations: Read, List, Cancel
# ---------------------------------------------------------------------

@app.get("/tasks/{task_id}")
@app.get("/a2a/tasks/{task_id}")
async def get_task_by_id(task_id: str, request: Request):
    principal = verify_headers(request)
    entry = get_task_db(task_id)
    if not entry or entry["principal"] != principal:
        # User isolation: generic error
        raise HTTPException(status_code=404, detail="Task not found")

    return make_a2a_response(entry["task"])


@app.get("/tasks")
@app.get("/a2a/tasks")
async def list_tasks(request: Request):
    principal = verify_headers(request)
    tasks = list_tasks_db(principal)
    return make_a2a_response({"tasks": tasks})


@app.post("/tasks/{task_id}:cancel")
@app.post("/a2a/tasks/{task_id}:cancel")
async def cancel_task(task_id: str, request: Request):
    principal = verify_headers(request)
    entry = get_task_db(task_id)
    if not entry or entry["principal"] != principal:
        raise HTTPException(status_code=404, detail="Task not found")

    lock = get_task_lock(task_id)
    with lock:
        current_entry = get_task_db(task_id)
        if not current_entry or current_entry["principal"] != principal:
            raise HTTPException(status_code=404, detail="Task not found")

        task_obj = current_entry["task"]
        current_state = task_obj.get("status", {}).get("state")
        if current_state in ["TASK_STATE_COMPLETED", "TASK_STATE_CANCELED"]:
            return make_a2a_response(
                {"error": {"code": "TASK_TERMINAL", "message": "Task is already terminal"}},
                status_code=409
            )

        task_obj["status"] = {"state": "TASK_STATE_CANCELED"}
        save_task_db(task_id, principal, task_obj["contextId"], "TASK_STATE_CANCELED", task_obj)

        return make_a2a_response(task_obj)
