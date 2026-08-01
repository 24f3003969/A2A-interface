import json
import os
import sys
import unittest
from fastapi.testclient import TestClient

# Ensure environment variables for testing
os.environ["DB_FILE"] = "test_storage.db"

if os.path.exists("test_storage.db"):
    os.remove("test_storage.db")

from app import app, init_db

init_db()
client = TestClient(app)

class TestA2AInvoiceAgent(unittest.TestCase):

    def setUp(self):
        self.headers_p1 = {
            "Authorization": "Bearer token_user_1",
            "A2A-Version": "1.0",
            "Content-Type": "application/a2a+json"
        }
        self.headers_p2 = {
            "Authorization": "Bearer token_user_2",
            "A2A-Version": "1.0",
            "Content-Type": "application/a2a+json"
        }

    def test_01_agent_card(self):
        res = client.get("/.well-known/agent-card.json")
        self.assertEqual(res.status_code, 200)
        card = res.json()
        self.assertEqual(card["name"], "Invoice Action Agent")
        self.assertIn("invoice_action_agent", [s["id"] for s in card["skills"]])

    def test_02_auth_and_version_and_media_type(self):
        # Missing auth -> 401
        res = client.post("/a2a/message:send", json={}, headers={"Content-Type": "application/a2a+json"})
        self.assertEqual(res.status_code, 401)
        self.assertEqual(res.headers["content-type"], "application/a2a+json")

        # Invalid version -> 400
        res = client.post("/a2a/message:send", headers={"Authorization": "Bearer token_user_1", "A2A-Version": "2.0", "Content-Type": "application/a2a+json"}, json={})
        self.assertEqual(res.status_code, 400)

        # Invalid Media Type -> 415
        res = client.post("/a2a/message:send", headers={"Authorization": "Bearer token_user_1", "A2A-Version": "1.0", "Content-Type": "application/json"}, json={})
        self.assertEqual(res.status_code, 415)

    def test_03_initial_batch_and_proposals(self):
        payload = {
            "message": {
                "messageId": "msg-001",
                "role": "ROLE_USER",
                "parts": [{
                    "mediaType": "application/vnd.ga5.invoice-claim-batch+json",
                    "data": {
                        "batchId": "batch-101",
                        "policyRevision": "v1.0",
                        "packages": [
                            {
                                "packageId": "pkg-001",
                                "text": "Invoice from Acme Corp [REF-101] [REF-102] [REF-103]. Total amount INR 5000. Valid invoice ready for settlement.",
                                "facts": {"vendorName": "Acme Corp", "invoiceNumber": "INV-101", "amountMinor": 500000, "currency": "INR"}
                            },
                            {
                                "packageId": "pkg-002",
                                "text": "Duplicate invoice found for Beta Inc [EVD-201] [EVD-202] [EVD-203]. Already paid under previous record.",
                                "facts": {"vendorName": "Beta Inc", "invoiceNumber": "INV-202", "amountMinor": 12000, "currency": "INR"}
                            }
                        ]
                    }
                }]
            },
            "configuration": {
                "returnImmediately": False,
                "historyLength": 20,
                "acceptedOutputModes": [
                    "application/vnd.ga5.invoice-action-proposals+json",
                    "application/vnd.ga5.invoice-action-receipts+json"
                ]
            }
        }

        res = client.post("/a2a/message:send", headers=self.headers_p1, json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("task", data)
        task = data["task"]
        self.assertEqual(task["status"]["state"], "TASK_STATE_INPUT_REQUIRED")
        self.assertEqual(len(task["artifacts"]), 1)

        proposals = task["artifacts"][0]["parts"][0]["data"]["proposals"]
        self.assertEqual(len(proposals), 2)
        for p in proposals:
            self.assertIn("packageId", p)
            self.assertIn("actionId", p)
            self.assertGreaterEqual(len(p["actionId"]), 12)
            self.assertIn("action", p)
            self.assertIn("facts", p)
            self.assertEqual(len(p["evidenceRefs"]), 3)
            self.assertGreaterEqual(len(p["rationale"]), 60)
            self.assertLessEqual(len(p["rationale"]), 1500)

        TestA2AInvoiceAgent.task_id = task["id"]
        TestA2AInvoiceAgent.context_id = task["contextId"]
        TestA2AInvoiceAgent.proposals = proposals

    def test_04_idempotency_and_persistent_replay(self):
        payload = {
            "message": {
                "messageId": "msg-idemp-001",
                "role": "ROLE_USER",
                "parts": [{
                    "mediaType": "application/vnd.ga5.invoice-claim-batch+json",
                    "data": {
                        "batchId": "batch-idemp-101",
                        "policyRevision": "v1.0",
                        "packages": [
                            {"packageId": "pkg-idemp-001", "text": "Invoice text [REF-1] [REF-2] [REF-3]"}
                        ]
                    }
                }]
            }
        }

        res1 = client.post("/a2a/message:send", headers=self.headers_p1, json=payload)
        self.assertEqual(res1.status_code, 200)

        # Replay initial message -> returns 200
        res2 = client.post("/a2a/message:send", headers=self.headers_p1, json=payload)
        self.assertEqual(res2.status_code, 200)
        self.assertEqual(res1.json(), res2.json())

        # Reused messageId with DIFFERENT content -> 409 Conflict
        payload_diff = json.loads(json.dumps(payload))
        payload_diff["message"]["parts"][0]["data"]["batchId"] = "different-batch"

        res_conflict = client.post("/a2a/message:send", headers=self.headers_p1, json=payload_diff)
        self.assertEqual(res_conflict.status_code, 409)
        self.assertEqual(res_conflict.json()["error"]["code"], "IDEMPOTENCY_CONFLICT")

    def test_05_continuation_and_persistent_replay(self):
        p1 = self.proposals[0]
        p2 = self.proposals[1]

        continuation_payload = {
            "message": {
                "messageId": "msg-002",
                "taskId": self.task_id,
                "contextId": self.context_id,
                "role": "ROLE_USER",
                "parts": [{
                    "mediaType": "application/vnd.ga5.invoice-action-results+json",
                    "data": {
                        "batchId": "batch-101",
                        "results": [
                            {
                                "packageId": p1["packageId"],
                                "actionId": p1["actionId"],
                                "action": p1["action"],
                                "outcome": "ACCEPTED",
                                "receiptNonce": "nonce-p1-12345"
                            },
                            {
                                "packageId": p2["packageId"],
                                "actionId": p2["actionId"],
                                "action": p2["action"],
                                "outcome": "REJECTED"
                            }
                        ]
                    }
                }]
            }
        }

        res = client.post("/a2a/message:send", headers=self.headers_p1, json=continuation_payload)
        self.assertEqual(res.status_code, 200)
        task = res.json()["task"]
        self.assertEqual(task["status"]["state"], "TASK_STATE_COMPLETED")
        self.assertEqual(len(task["artifacts"]), 2)

        # Persistent Replay Check: Replaying initial message msg-001 now MUST return the terminal COMPLETED task!
        initial_payload = {
            "message": {
                "messageId": "msg-001",
                "role": "ROLE_USER",
                "parts": [{
                    "mediaType": "application/vnd.ga5.invoice-claim-batch+json",
                    "data": {
                        "batchId": "batch-101",
                        "policyRevision": "v1.0",
                        "packages": [
                            {
                                "packageId": "pkg-001",
                                "text": "Invoice from Acme Corp [REF-101] [REF-102] [REF-103]. Total amount INR 5000. Valid invoice ready for settlement.",
                                "facts": {"vendorName": "Acme Corp", "invoiceNumber": "INV-101", "amountMinor": 500000, "currency": "INR"}
                            },
                            {
                                "packageId": "pkg-002",
                                "text": "Duplicate invoice found for Beta Inc [EVD-201] [EVD-202] [EVD-203]. Already paid under previous record.",
                                "facts": {"vendorName": "Beta Inc", "invoiceNumber": "INV-202", "amountMinor": 12000, "currency": "INR"}
                            }
                        ]
                    }
                }]
            }
        }
        res_replay = client.post("/a2a/message:send", headers=self.headers_p1, json=initial_payload)
        self.assertEqual(res_replay.status_code, 200)
        self.assertEqual(res_replay.json()["task"]["status"]["state"], "TASK_STATE_COMPLETED")

    def test_06_user_isolation(self):
        res = client.get(f"/a2a/tasks/{self.task_id}", headers=self.headers_p2)
        self.assertEqual(res.status_code, 404)
        self.assertEqual(res.headers["content-type"], "application/a2a+json")

        res_list = client.get("/a2a/tasks", headers=self.headers_p2)
        self.assertEqual(res_list.status_code, 200)
        self.assertEqual(res_list.json()["tasks"], [])


if __name__ == "__main__":
    unittest.main()
