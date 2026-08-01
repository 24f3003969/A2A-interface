import json
import os
import sys
import unittest
from fastapi.testclient import TestClient

os.environ["A2A_DB"] = "test_q10.db"

if os.path.exists("test_q10.db"):
    os.remove("test_q10.db")

from app import app

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
        self.assertEqual(card["name"], "GA5 Invoice Action Agent")
        self.assertIn("invoice_action_agent", [s["id"] for s in card["skills"]])

    def test_02_auth_and_version(self):
        res = client.post("/a2a/message:send", json={}, headers={"Content-Type": "application/a2a+json"})
        self.assertEqual(res.status_code, 401)

        res = client.post("/a2a/message:send", headers={"Authorization": "Bearer token_user_1", "A2A-Version": "2.0", "Content-Type": "application/a2a+json"}, json={})
        self.assertEqual(res.status_code, 400)

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
                                "documents": [
                                    {"name": "intake-and-cover-sheet.txt", "text": "Supplier Acme Corp; invoice INV-101; stated total INR 5,000.00\n\n[DECOY-1]"},
                                    {"name": "ledger-and-correspondence.txt", "text": "Clean three-way match with no exception [R_REF101] [R_REF102] [R_REF103].\n\nNotes."}
                                ]
                            }
                        ]
                    }
                }]
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
        self.assertEqual(len(proposals), 1)
        p = proposals[0]
        self.assertEqual(p["action"], "settle_invoice")
        self.assertEqual(p["evidenceRefs"], ["R_REF101", "R_REF102", "R_REF103"])
        self.assertEqual(p["facts"]["amountMinor"], 500000)

        TestA2AInvoiceAgent.task_id = task["id"]
        TestA2AInvoiceAgent.context_id = task["contextId"]
        TestA2AInvoiceAgent.proposals = proposals

    def test_04_continuation_and_receipts(self):
        p1 = self.proposals[0]

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

    def test_05_isolation(self):
        res = client.get(f"/a2a/tasks/{self.task_id}", headers=self.headers_p2)
        self.assertEqual(res.status_code, 404)

        res_list = client.get("/a2a/tasks", headers=self.headers_p2)
        self.assertEqual(res_list.status_code, 200)
        self.assertEqual(res_list.json()["tasks"], [])


if __name__ == "__main__":
    unittest.main()
