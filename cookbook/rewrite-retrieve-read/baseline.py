from langchain.callbacks.tracers import ConsoleCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.utilities import DuckDuckGoSearchAPIWrapper

template = """Answer the users question based only on the following context:

<context>
{context}
</context>

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

model = ChatOpenAI(temperature=0)
search = DuckDuckGoSearchAPIWrapper()


def retriever(query):
    return search.run(query)


chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)

# simple_query = "what is langchain?"
distracted_query = (
    "man that sam bankman fried trial was crazy! what is langchain?"
)
print(
    chain.invoke(
        distracted_query, config={"callbacks": [ConsoleCallbackHandler()]}
    )
)
