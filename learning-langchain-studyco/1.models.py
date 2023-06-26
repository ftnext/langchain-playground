from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage

chat = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

result = chat([HumanMessage(content="GPTとして自己紹介してください。日本語で2文程度でお願いします")])
print(result.content)
