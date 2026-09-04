"""Manual live smoke-test client for Module 16 (not part of the test suite).

Connects to a live uvicorn instance and verifies, over a REAL socket:

    CONNECT           -> connected
    ping              -> pong
    prediction_request-> prediction_result  (REAL GNN model output)
    prediction_request-> single-node result (requested_node_id echo)
    ping              -> pong  (connection remains usable)
    disconnect

With a trained checkpoint configured (PREDICTION_CHECKPOINT_PATH) the success
path is a REAL ``prediction_result``; the client exits non-zero otherwise and
clearly distinguishes the infrastructure failure modes instead of hiding them:

    MODEL_UNAVAILABLE   -> no trained checkpoint configured / file missing
                           (run backend/scripts/train_gnn.py, then set
                           PREDICTION_CHECKPOINT_PATH in the root .env)
    PREDICTION_FAILED + "graph database" -> Neo4j is unreachable/down
                           (start Neo4j; seed demo data with
                           backend/scripts/seed_demo_graph.py if empty)
    PREDICTION_FAILED + other message  -> graph/model inference failure
                           (inspect the server logs)

Neo4j must be running and contain at least one entity for a REAL prediction:
the dataset is built from the live graph before inference.
"""

import asyncio
import json
import sys

import websockets

WS_URL = "ws://127.0.0.1:8765/api/v1/ws"


async def main() -> int:
    async with websockets.connect(WS_URL) as ws:
        # 1. CONNECT -> connected
        connected = json.loads(await ws.recv())
        print("CONNECT   ->", connected["type"], "supported:", connected["data"]["supported_client_messages"])
        assert connected["type"] == "connected"
        assert "prediction_request" in connected["data"]["supported_client_messages"]

        # 2. ping -> pong
        await ws.send(json.dumps({"type": "ping"}))
        pong = json.loads(await ws.recv())
        print("PING      ->", pong["type"])
        assert pong["type"] == "pong"

        # 3. prediction_request -> prediction_result (whole graph)
        await ws.send(json.dumps({"type": "prediction_request"}))
        response = json.loads(await ws.recv())
        if response["type"] != "prediction_result":
            _report_failure_and_exit(response)

        data = response["data"]
        count = data["prediction_count"]
        first = data["predictions"][0]
        print("PREDICTION_REQUEST -> prediction_result (REAL model output)")
        print(f"  prediction_count: {count}")
        print(f"  first entry: node_id={first['node_id']!r} prediction={first['prediction']}")
        assert count == len(data["predictions"])
        assert count >= 1

        # 4. single-node request selects ONLY that node from the real output
        node_id = first["node_id"]
        expected_value = first["prediction"]
        await ws.send(
            json.dumps({"type": "prediction_request", "data": {"node_id": node_id}})
        )
        single = json.loads(await ws.recv())
        if single["type"] != "prediction_result":
            _report_failure_and_exit(single)
        single_data = single["data"]
        print(f"SINGLE NODE  -> prediction_result for {single_data.get('requested_node_id')!r}")
        assert single_data.get("requested_node_id") == node_id
        assert single_data["prediction_count"] == 1
        assert single_data["predictions"][0]["node_id"] == node_id
        assert single_data["predictions"][0]["prediction"] == expected_value

        # 5. connection remains usable
        await ws.send(json.dumps({"type": "ping"}))
        pong2 = json.loads(await ws.recv())
        print("PING after prediction ->", pong2["type"])
        assert pong2["type"] == "pong"

    # 6. disconnect (exits the context manager)
    print("DISCONNECT -> ok")
    print("SMOKE TEST PASSED: real GNN prediction served over WebSocket")
    return 0


def _report_failure_and_exit(response: dict) -> None:
    error = response.get("error", {})
    code = error.get("code", "?")
    message = error.get("message", "?")
    print(f"PREDICTION_REQUEST -> error [{code}]: {message}")
    if code == "MODEL_UNAVAILABLE":
        print(
            "\nCause: no trained checkpoint configured (or file missing).\n"
            "Fix:   cd backend\n"
            "       python scripts/train_gnn.py --synthetic\n"
            "       then set PREDICTION_CHECKPOINT_PATH=checkpoints/gnn_m13.pt "
            "in the root .env and restart uvicorn."
        )
    elif code == "PREDICTION_FAILED" and "graph database" in message:
        print(
            "\nCause: Neo4j is unreachable (the checkpoint itself is fine).\n"
            "Fix:   start your Neo4j instance, and if the graph is empty run\n"
            "       cd backend && python scripts/seed_demo_graph.py"
        )
    else:
        print(
            "\nCause: graph/model inference failure. Check the uvicorn logs "
            "for the structured error details."
        )
    print("\nSMOKE TEST FAILED")
    sys.exit(1)


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
