import argparse

from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage

parser = argparse.ArgumentParser()
parser.add_argument("question")
args = parser.parse_args()

chat = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

result = chat([HumanMessage(content=args.question)])
print(result.content)
