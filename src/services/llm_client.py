import httpx
import logging
from config import REMOTE_LLM_URL, LLM_TIMEOUT

logger = logging.getLogger("LLMClient")

async def call_llm(prompt: str) -> str:
    """
    Sends a prompt to the remote LLM service (Colab + ngrok).
    Expected response format: { "status": "ok", "response": "..." }
    """
    payload = { "prompt": prompt }
    
    try:
        logger.info(f"[LLMClient] Sending request to {REMOTE_LLM_URL}")
        async with httpx.AsyncClient(timeout=LLM_TIMEOUT) as client:
            response = await client.post(REMOTE_LLM_URL, json=payload)
            response.raise_for_status()
            
            data = response.json()
            if data.get("status") != "ok":
                raise Exception(f"Remote LLM error: {data.get('response', 'Unknown error')}")
                
            return data.get("response", "").strip()
            
    except httpx.TimeoutException:
        logger.error("[LLMClient] Remote LLM timed out")
        raise Exception("LLM service timed out")
    except Exception as e:
        logger.error(f"[LLMClient] Remote LLM error: {str(e)}")
        raise e
