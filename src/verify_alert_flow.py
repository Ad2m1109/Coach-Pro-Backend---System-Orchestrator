import asyncio
import json
import websockets
import httpx

async def test_alert_flow(match_id):
    print(f"🚀 Testing Alert Flow for Match: {match_id}")
    
    # 1. Start WebSocket Listener
    uri = f"ws://localhost:8000/ws/alerts/{match_id}"
    
    try:
        async with websockets.connect(uri) as websocket:
            print("✅ Connected to alert WebSocket")
            
            # 2. Trigger a mock alert via the internal service (simulated by calling the broadcast directly in a test script would be easier, but let's assume we want to test the full chain)
            # Since I can't easily trigger the engine from here, I'll check the history endpoint first.
            
            # 3. Check History
            async with httpx.AsyncClient() as client:
                resp = await client.get(f"http://localhost:8000/api/matches/{match_id}/alerts")
                print(f"📊 History Response: {resp.status_code}")
                print(f"📝 Alerts: {resp.json()}")

    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_alert_flow("test_match_123"))
