"""Rewrite-Retrieve-Read"""
# https://github.com/langchain-ai/langchain/blob/v0.0.339/cookbook/rewrite.ipynb
# pip install langchain openai duckduckgo-search

from langchain import hub
from langchain.callbacks.tracers import ConsoleCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.utilities import DuckDuckGoSearchAPIWrapper

template = """Provide a better search query for web search engine \
to answer the given question, end the queries with ' **' . \
Question: {x} Answer:"""
rewrite_prompt = ChatPromptTemplate.from_template(template)
# rewrite_prompt = hub.pull("langchain-ai/rewrite")


def _parse(text):
    return text.strip("**")


rewriter = (
    rewrite_prompt | ChatOpenAI(temperature=0) | StrOutputParser() | _parse
)

distracted_query = (
    "man that sam bankman fried trial was crazy! what is langchain?"
)
# rewriter.invoke({"x": distracted_query})

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


rewrite_retrieve_read_chain = (
    {
        "context": {"x": RunnablePassthrough()} | rewriter | retriever,
        "question": RunnablePassthrough(),
    }
    | prompt
    | model
    | StrOutputParser()
)
print(
    rewrite_retrieve_read_chain.invoke(
        distracted_query, config={"callbacks": [ConsoleCallbackHandler()]}
    )
)
