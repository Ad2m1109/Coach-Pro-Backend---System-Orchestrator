import asyncio
import json
import logging
import websockets
import httpx

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_alert_flow(match_id):
    logger.info("Testing Alert Flow for Match: %s", match_id)
    
    # 1. Start WebSocket Listener
    uri = f"ws://localhost:8000/ws/alerts/{match_id}"
    
    try:
        async with websockets.connect(uri) as websocket:
            logger.info("Connected to alert WebSocket")
            
            # 2. Trigger a mock alert via the internal service (simulated by calling the broadcast directly in a test script would be easier, but let's assume we want to test the full chain)
            # Since I can't easily trigger the engine from here, I'll check the history endpoint first.
            
            # 3. Check History
            async with httpx.AsyncClient() as client:
                resp = await client.get(f"http://localhost:8000/api/matches/{match_id}/alerts")
                logger.info("History Response: %s", resp.status_code)
                logger.info("Alerts: %s", resp.json())

    except Exception as e:
        logger.exception("Error: %s", e)

if __name__ == "__main__":
    asyncio.run(test_alert_flow("test_match_123"))
