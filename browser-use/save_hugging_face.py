# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = [
#     "browser-use",
#     "langchain-openai",
#     "pydantic",
# ]
# ///
# ref: https://github.com/browser-use/browser-use/blob/a30e17c40c8e7c3c9e88f29ebc2d8186794254de/examples/save_to_file_hugging_face.py
# TODO: langchain-google-genai (HTML content parse error)
import os

os.environ["ANONYMIZED_TELEMETRY"] = "false"

import asyncio

from browser_use import Agent, Controller
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

controller = Controller()


class Model(BaseModel):
    name: str
    url: str
    likes: int
    license: str

    def __str__(self):
        return f"{self.name} ({self.url}): {self.likes} likes, {self.license}"


class Models(BaseModel):
    models: list[Model]

    def __iter__(self):
        return iter(self.models)


@controller.action("Save models", param_model=Models)
def save_models(params: Models):
    with open("models.txt", "w") as f:
        for model in params:
            f.write(f"{model}\n")


async def main():
    agent = Agent(
        task="Look up models with a license of cc-by-sa-4.0 and sort by most likes on Hugging face, save top 5 to file.",
        llm=ChatOpenAI(model="gpt-4o"),
        controller=controller,
    )
    return await agent.run()


if __name__ == "__main__":
    print(asyncio.run(main()))
