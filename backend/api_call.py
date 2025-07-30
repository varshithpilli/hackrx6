import httpx
import json

async def api_call(self, messages, max_retries=10):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": 1024
        }
        
        print(headers)

        timeout = httpx.Timeout(connect=10, read=120, write=10, pool=5)

        for attempt in range(max_retries):
            try:
                async with httpx.AsyncClient(timeout=timeout) as client:
                    response = await client.post(self.base_url, headers=headers, json=payload)

                data = response.json()

                if "choices" in data:
                    return data["choices"][0]["message"]["content"]

                print("[ERROR] LLM API Response:", json.dumps(data, indent=2))
            except httpx.ReadTimeout:
                print(f"[Retry {attempt+1}] Request timed out, retrying...")

        return "Error: LLM API timed out after multiple retries"
    
    