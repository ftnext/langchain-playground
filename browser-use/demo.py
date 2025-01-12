# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = [
#     "browser-use",
#     "langchain-google-genai",
# ]
# ///
import os
os.environ["ANONYMIZED_TELEMETRY"] = "false"

import asyncio

from browser_use import Agent
from langchain_google_genai import ChatGoogleGenerativeAI


async def main():
    agent = Agent(
        task="Find a one-way flight from Bali to Oman on 19 January 2025 on Google Flights. Return me the cheapest option.",
        llm=ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp"),
    )
    result = await agent.run()
    print(result)


asyncio.run(main())
