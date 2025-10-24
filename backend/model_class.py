import asyncio
import os
from dotenv import load_dotenv
from .infere import get_chunks
from .utils import log_time
import time
from .api_call import open_api_call, gemini_api_call, deepseek_api_call

load_dotenv()

API_KEY = os.getenv("API_KEY")
BASE_URL = os.getenv("BASE_URL")
MODEL = os.getenv("MODEL")

class Constitutioner:
    def __init__(self):
        self.api_key = API_KEY
        self.base_url = BASE_URL
        self.model = MODEL

    def system_prompt(self):
        return """
            You are ClauseGPT, an AI assistant that helps users understand and analyze documents related to insurance policies, legal contracts, HR policies, and compliance guidelines.

            Your goals:
            - Provide clear, fact-based answers to user questions based strictly on the document content.
            - Use simple, plain language that is easy for non-technical and non-legal users to understand.

            Instructions:
            1. Your response must stay grounded in the document. Do not add or assume anything not present in the document.
            2. Write in plain text using full sentences. Avoid Markdown formatting and symbols like asterisks, bullet points, or bold text.
            3. Do not start the response with phrases like "Answer:", just respond directly and naturally.
            4. Keep your response concise, direct, and focused on facts from the document.
            5. Use basic terms like premium, grace period, or waiting period if needed, and explain them simply if they appear in the document.

            Tone:
            - Clear and neutral.
            - Avoid legal or technical jargon unless it is explicitly in the document and necessary for accuracy.
            - Prioritize ease of understanding for everyday users.

        """

    def user_prompt(self, query, docs):
        context = "\n\n".join([doc['content'] for doc in docs])

        return f"""Based on the following Official Document snippets, answer the user's query:
            CONTEXT:
            {context}

            USER QUERY:
            {query}
        """        
            
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
        return await deepseek_api_call(messages)

    # async def run_parallel_inference(model, queries, chunks, embeddings):
    #     tasks = [model.inference(query, chunks, embeddings) for query in queries]
    #     results = await asyncio.gather(*tasks)
    #     return results
