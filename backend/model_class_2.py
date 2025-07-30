import httpx
import os
from dotenv import load_dotenv
from openai import OpenAI
from .infere import get_chunks
import json
from .utils import log_time
import time

load_dotenv()

API_KEY = os.getenv("API_KEY")
BASE_URL = os.getenv("BASE_URL")
MODEL = os.getenv("MODEL")

class Constitutioner:
    def __init__(self):
        self.api_key = "sk-72a20befadd045fd9d5c55d4e8ae4aca"
        self.base_url = "https://api.deepseek.com"
        self.model = MODEL

    def system_prompt(self):
        return """
            You are **ClauseGPT**, an AI assistant that helps users understand and analyze documents
            related to **insurance policies, legal contracts, HR policies, and compliance guidelines**.

            Your goals:
            - Provide **clear, fact-based answers** to user questions based on the document content.
            - Use **plain language** that is easy for non-technical and non-legal users to understand.

            ---
            ### When answering:

            1. **Stay grounded in the document content.**

            2. **Never make up details not found in the document. You can only generate data from what the document provides.**

            3. **Output format:** Provide a short **answer summary** and also be precise and concise.

            ---
            ### Example Response Style:

            **Answer:**  
            Yes, this policy covers maternity expenses. However, it is only available if the insured
            person has been continuously covered for at least 24 months, and the benefit is limited
            to two deliveries during the policy period.

            ---
            ### Tone:
            - Be **clear, concise, and factual**.
            - Avoid unnecessary legal jargon, but use terms like *premium, grace period, waiting period*
            when necessary and explain them briefly.

        """

    def user_prompt(self, query, docs):
        context = "\n\n".join([doc['content'] for doc in docs])

        return f"""Based on the following official Official Document snippets, answer the user's query:
            CONTEXT:
            {context}

            USER QUERY:
            {query}
        """        

    def api_call(self, messages):
        client = OpenAI(api_key=self.api_key, base_url=self.base_url)

        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            stream=False
        )

        return response.choices[0].message.content

    
    # async def api_call(self, messages, max_retries=3):
    #     headers = {
    #         "Authorization": f"Bearer {self.api_key}",
    #         "Content-Type": "application/json"
    #     }
    #     payload = {
    #         "model": self.model,
    #         "messages": messages,
    #         "max_tokens": 1024
    #     }

    #     timeout = httpx.Timeout(connect=10, read=120, write=10, pool=5)

    #     for attempt in range(max_retries):
    #         try:
    #             async with httpx.AsyncClient(timeout=timeout) as client:
    #                 response = await client.post(self.base_url, headers=headers, json=payload)

    #             data = response.json()

    #             if "choices" in data:
    #                 return data["choices"][0]["message"]["content"]

    #             print("[ERROR] LLM API Response:", json.dumps(data, indent=2))
    #         except httpx.ReadTimeout:
    #             print(f"[Retry {attempt+1}] Request timed out, retrying...")

    #     return "Error: LLM API timed out after multiple retries"
    
    
    async def inference(self, query, chunks, embeddings):
        start = time.perf_counter()

        docs = get_chunks(query, chunks, embeddings)

        end = time.perf_counter()
        log_time("get_chunks", start, end)

        messages = [
            {"role": "system", "content": self.system_prompt()},
            {"role": "user", "content": self.user_prompt(query, docs)}
        ]

        # print(messages[1]['content'])
        # return await self.api_call(messages)
        return self.api_call(messages)