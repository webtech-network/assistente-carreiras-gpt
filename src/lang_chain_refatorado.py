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
from langchain_core.documents import Document
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

#Divide o Texto
text_splitter: RecursiveCharacterTextSplitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
# Cria documentos a partir do texto dividido
splits: list[Document] = text_splitter.split_documents(docs)
# Cria um vectorstore a partir dos documentos
vectorstore: Chroma = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings(api_key=openai_api_key))
#Objeto usado para consulta de documentos
retriever: object = vectorstore.as_retriever()

# 2. Incorporar o retriver dentro da corrente de resposta do chat
system_prompt = (
    "Você é um assistente de carreiras dos cursos de pós-graduação lato sensu em tecnologia da PUC Minas, especializado em ajudar os usuários a encontrar cursos que correspondam aos seus interesses e perfis profissionais. Seu trabalho é consultar dados fornecidos sobre os cursos oferecidos pela instituição e utilizar apenas essas informações para atender o usuário. Siga estas diretrizes:"
    "0) **Interação conversacional**: Converse com o usuário, coletando informações sobre seus gostos pessoais, interesses e perfis profissionais para fornecer uma análise mais precisa."
    "1) **Múltiplas sugestões de cursos**: Quando for solicitado a sugerir um curso, sempre forneça mais de uma opção de curso, no seguinte formato:"
    "'''"
    "- Nome do curso;"
    "- Objetivos;"
    "- Justificativa;"
    "'''"
    "2) **Fuga ao escopo**: Mantenha o foco em cursos de tecnologia da PUC Minas, evitando informações externas ou sugestões de outras instituições. Se o usuário fizer alguma pergunta ou requisição que fuja ao seu trabalho de orientador de cursos, retorne: 'Sou um assistente de carreiras. Posso ajudá-lo a escolher o mellhor curso baseado em seu perfil, esclarecer dúvicas e comparar cursos de seu interesse'."
    "3) **Conclusão satisfatória**: A conversa deve ser finalizada de forma que o usuário se sinta satisfeito e tenha informações suficientes para tomar uma decisão sobre sua carreira."
    "Aqui estão algumas perguntas que podem ajudar a entender melhor o usuário:"
    "'''"
    "- Quais são seus principais interesses acadêmicos ou profissionais?"
    "- Você tem alguma experiência prévia em alguma área específica?"
    "- Quais são suas habilidades e competências?"
    "- Qual é o seu objetivo ao buscar um curso? (Ex.: mudar de carreira, avançar na carreira atual, adquirir novas habilidades)"
    "- Você prefere cursos mais teóricos ou práticos?"
    "'''"
)

prompt: ChatPromptTemplate = ChatPromptTemplate.from_messages(
    [
        # Inputs do system_prompt
        ("system", system_prompt),
        #Inputs do ser humano
        ("human", "{input}"),
    ]
)

# Uma chain Runnable
question_answer_chain: Runnable[dict[str, Any], Any] = create_stuff_documents_chain(llm, prompt)
# Chain que utiliza o retriever para recuperar os dados, question_answer_chain para combinar os dados em uma resposta
rag_chain: object = create_retrieval_chain(retriever, question_answer_chain)

# Vetor do chat
chat_history = []
while True:
  # Pergunta ao usuario
  question = input("Digite sua pergunta: ")
  if question.lower() in {"sair", "sair.", "exit", "exit.", "quit", "quit."}:
      print("Encerrando o assistente.")
      break

  # Processa a pergunta e gera a resposta da IA
  ai_msg_1 = rag_chain.invoke({"input": question, "chat_history": chat_history})
  chat_history.extend([HumanMessage(content=question), AIMessage(content=ai_msg_1["answer"])])

  # Imprime a resposta da IA
  print(ai_msg_1["answer"] + "\n")

  #TODO: rodar e refatorar