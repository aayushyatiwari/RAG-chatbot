from PyPDF2 import PdfReader
import chromadb
from langchain_openai import OpenAIEmbeddings
import os
from typing import Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field
from typing_extensions import TypedDict
from sentence_transformers import SentenceTransformer
import getpass
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
# from dotenv import load_dotenv
# load_dotenv()
from google.auth import exceptions
from groq import Groq
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch





os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")

#  Set BOTH environment variables (some internal checks need GOOGLE_API_KEY)
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
# os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
# GROQ_API_KEY =os.getenv("GROQ_API_KEY")

# llm = ChatGroq(
#     model="llama-3.3-70b-versatile",
#     temperature=0.7,
# )

class State(TypedDict):
    messages : Annotated[list, add_messages]

graph_builder = StateGraph(State)



os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
llm = init_chat_model("google_genai:gemini-2.0-flash")


def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}

graph_builder.add_node("chatbot", chatbot)

graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)
graph = graph_builder.compile()

user_input = input("YOU: ")
state = graph.invoke({"messages": [{"role":"user", "content": user_input}]})
print("BOT: ", state["messages"][-1].content)
