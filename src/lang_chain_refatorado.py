#%%
#! pip install langchain langchain_chroma langchain_core langchain_text_splitters langchain_openai langchain_community
#%%
from typing import Dict, Any, List, Union
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.chains import create_history_aware_retriever
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import Runnable
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader

# Caso não seja possivel colocar a chave nas variaveis de ambiente insira manualmente aqui
openai_api_key: str ="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"

llm: ChatOpenAI = ChatOpenAI(
    model="gpt-4o",
    temperature=0.5, # A temperatura varia de 0 ate 1
    max_tokens=None,
    api_key=openai_api_key,
    # timeout=None,
    # max_retries=None
)

# Carrega o pdf
loader: PyPDFLoader = PyPDFLoader("assistente-carreiras-gpt/assets/DATA/Cursos_completos.pdf", extract_images=False)
docs: List[Dict[str, Any]] = loader.load()

