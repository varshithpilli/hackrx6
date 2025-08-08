import httpx
import json
import google.generativeai as genai
import time
from .utils import log_time
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("API_KEY")
deepseek_api_key = os.getenv("DEEPSEEK_API")
gemini_api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=gemini_api_key)

async def gemini_api_call(messages):
    print("LLM request started")
    start = time.perf_counter()
    model = genai.GenerativeModel("gemini-2.0-flash")
    prompt = f"Answer the following question based on the attached document:\n\n{messages}"

    response = await model.generate_content_async(prompt)

    answer = response.text.strip()

    end = time.perf_counter()
    log_time("LLM api call", start, end)
    print("LLM request ended")
    return answer

async def open_api_call(messages, max_retries=10):
        print("LLM request started")
        start = time.perf_counter()
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "deepseek/deepseek-r1-0528-qwen3-8b:free",
            "messages": messages,
            "max_tokens": 1024
        }
        
        print(headers)

        timeout = httpx.Timeout(connect=10, read=120, write=10, pool=5)

        for attempt in range(max_retries):
            try:
                async with httpx.AsyncClient(timeout=timeout) as client:
                    response = await client.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload)

                data = response.json()

                if "choices" in data:
                    end = time.perf_counter()
                    log_time("LLM api call", start, end)
                    print("LLM request ended")
                    return data["choices"][0]["message"]["content"]

                print("[ERROR] LLM API Response:", json.dumps(data, indent=2))
            except httpx.ReadTimeout:
                print(f"[Retry {attempt+1}] Request timed out, retrying...")

        return "Error: LLM API timed out after multiple retries"



async def deepseek_api_call(messages, max_retries=10):
    print("LLM request started")
    start = time.perf_counter()

    headers = {
        "Authorization": f"Bearer {deepseek_api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "deepseek-chat",
        "messages": messages,
        "max_tokens": 1024
    }

    timeout = httpx.Timeout(connect=10, read=120, write=10, pool=5)

    for attempt in range(max_retries):
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post("https://api.deepseek.com/v1/chat/completions", headers=headers, json=payload)
            
            data = response.json()

            if "choices" in data:
                end = time.perf_counter()
                log_time("Deepseek api call", start, end)
                print("LLM request ended")
                return data["choices"][0]["message"]["content"]

            print("[ERROR] LLM API Response:", json.dumps(data, indent=2))
        except httpx.ReadTimeout:
            print(f"[Retry {attempt+1}] Request timed out, retrying...")
        except Exception as e:
            print(f"[Retry {attempt+1}] Unexpected error: {e}")

    return "Error: LLM API timed out after multiple retries"
