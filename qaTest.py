import os
from collections import deque
from typing import Dict, List, Optional, Any
import langchain 
import openai 
import pinecone 
from langchain.chains import LLMChain
from langchain.chains.base import Chain
from langchain.agents import AgentType, ZeroShotAgent, Tool, AgentExecutor, initialize_agent
from langchain.llms import OpenAI, LlamaCpp, BaseLLM
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory, ReadOnlySharedMemory
from langchain.utilities import GoogleSearchAPIWrapper
from langchain.vectorstores.base import VectorStore
from pydantic import BaseModel, Field
from langchain.embeddings import OpenAIEmbeddings
import faiss
from langchain.vectorstores import FAISS
from langchain.docstore import InMemoryDocstore

def execute_agent(input: str):
    template = """This is a conversation between a human and a bot:

        {chat_history}

        Write a summary of the conversation for {input}:
    """

    llm = OpenAI(temperature=0)

    prompt = PromptTemplate(template=template, input_variables=["chat_history", "input"])

    memory = ConversationBufferMemory(memory_key="chat_history") 

    read_memory = ReadOnlySharedMemory(memory=memory)

    summary_chain = LLMChain(memory=read_memory, prompt=prompt, llm=llm, verbose=True)

    search = GoogleSearchAPIWrapper()

    tools = [
        Tool(
            name = "Search",
            func=search.run,
            description="useful for when you need to answer questions about current events"
        ),
        Tool(
            name = "Summary",
            func=summary_chain.run,
            description="useful for when you summarize a conversation. The input to this tool should be a string, representing who will read this summary."
        )
    ]   

    prefix = """Please have a conversation with a being, answering their following questions as best you can. You have access to the following tools:"""
    suffix = """Begin!"

    {chat_history}
    Question: {input}
    {agent_scratchpad}"""

    prompt2 = ZeroShotAgent.create_prompt(
        tools, 
        prefix=prefix, 
        suffix=suffix, 
        input_variables=["input", "chat_history", "agent_scratchpad"]
    )

    llm_chain = LLMChain(llm=OpenAI(temperature=0), prompt=prompt2)

    agent = ZeroShotAgent(llm_chain=llm_chain, tools=tools, verbose=True)

    agent_chain = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True, memory=memory)

    agent_chain.run(input=input)
    agent_chain.run(input="Based on the previous answer, what are some of their achievements?")


def main(): 
    execute_agent("Who is the current CEO of OpenAI?")

main()