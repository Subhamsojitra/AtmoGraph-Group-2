"""Temporary Module 16 manual smoke-test client (not part of the test suite).

Connects to a live uvicorn instance and verifies, over a REAL socket:

    CONNECT -> connected
    ping    -> pong
    prediction_request -> prediction_result OR structured MODEL_UNAVAILABLE error
    ping    -> pong  (connection remains usable)
    disconnect

No Neo4j is required: with no trained checkpoint configured the prediction
path legitimately answers `MODEL_UNAVAILABLE`, which proves the full
transport -> validation -> PredictionService dispatch works end to end.
"""

import asyncio
import json

import websockets

WS_URL = "ws://127.0.0.1:8765/api/v1/ws"


async def main() -> None:
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

        # 3. prediction_request -> prediction_result OR structured model-unavailable error
        await ws.send(
            json.dumps(
                {
                    "type": "prediction_request",
                    "data": {"node_id": "supplier-001"},
                }
            )
        )
        response = json.loads(await ws.recv())
        print("PREDICTION_REQUEST ->", response["type"], response.get("error", {}).get("code"))
        if response["type"] == "prediction_result":
            print("  predictions:", response["data"])
        else:
            assert response["type"] == "error"
            assert response["error"]["code"] == "MODEL_UNAVAILABLE"
            print("  (expected: no trained checkpoint is configured)")

        # 4. connection remains usable
        await ws.send(json.dumps({"type": "ping"}))
        pong2 = json.loads(await ws.recv())
        print("PING after prediction ->", pong2["type"])
        assert pong2["type"] == "pong"

    # 5. disconnect (exits the context manager)
    print("DISCONNECT -> ok")
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    asyncio.run(main())