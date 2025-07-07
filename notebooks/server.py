#!/usr/bin/env python

from fastapi import FastAPI
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from typing import Any

from langserve import add_routes
from pydantic import BaseModel
from langchain_community.document_loaders import WebBaseLoader
from langchain import hub
from langchain.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool
from langchain.agents import AgentExecutor, create_react_agent

class Input(BaseModel):
    input: str


class Output(BaseModel):
    output: Any

def cut_output(output):
    return output['output']


loader = WebBaseLoader("https://allopizza.su/spb/kupchino/about")
data = loader.load()

course_api_key = ''# ключ курса (если используем ключи из курса)
llm = ChatOpenAI(api_key=course_api_key, model='gpt-4o-mini',  
                 base_url="https://aleron-llm.neuraldeep.tech/")

embeddings = OpenAIEmbeddings(api_key=course_api_key, model='text-embedding-3-small', 
                              base_url="https://aleron-llm.neuraldeep.tech/")

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(data)
db_embed = FAISS.from_documents(texts, embeddings)
retriever = db_embed.as_retriever()

tool = create_retriever_tool(
    retriever, # наш ретривер
    "search_web", # имя инструмента
    "Searches and returns data from page", # описание инструмента подается в ЛЛМ
)

prompt = hub.pull("hwchase17/react")

agent = create_react_agent(llm, [tool], prompt)
agent_executor = AgentExecutor(agent=agent, tools=[tool])

app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="A simple api server using Langchain's Runnable interfaces",
)

add_routes(
    app,
    agent_executor.with_types(input_type=Input, output_type=Output) | cut_output,
    path="/rag_agent",  # эндпоинт для rag агента
)

prompt2 = ChatPromptTemplate.from_template("tell me a joke about {topic}")
add_routes(
    app,
    prompt2 | llm,
    path="/joke",  # эндпоинт для цепочки
)



if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8501)
