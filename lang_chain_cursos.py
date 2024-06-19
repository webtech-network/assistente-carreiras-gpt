# pip install langchain langchain_chroma langchain_core langchain_text_splitters langchain_openai langchain_community
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.chains import create_history_aware_retriever
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader

# Caso não seja possivel colocar a chave nas variaveis de ambiente incira manualmente aqui
# Senao deixe vazio
openai_api_key=""

llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0.7, # A temperatura varia de 0 ate 1
    max_tokens=None,
    api_key=openai_api_key,
    # timeout=None,
    # max_retries=2
    )

# 1. Carregar dividir e indexar o conteudo do arquivo
print("Inicializando")


loader = PyPDFLoader("DATA/Cursos_completos.pdf", extract_images=False)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings(api_key=openai_api_key))
retriever = vectorstore.as_retriever()

# 2. Incorporar o retriver dentro da corrente de resposta do chat

system_prompt = (
    "Você é um assistente que auxilia pessoas a acharem o curso ideal para elas"
    "O dados dcursos estao contidos no pdf qu elhe foi fornecido"
    "Você deve utilizar como base APENAS OS CURSOS DO PDF"
    "E não deve respoonder nada alem do que uma assistente de carreiras saberia"
    "RESTRINJA-SE APENAS AO QUE STA NO DOCUMENTO, NÃO TIRE NADA POR FORA"
    "Você esta a serviso da universidade PUC minas (Pontificia Univercidade Catolica)"
    "\n\n"
    "{context}"
)

# system_prompt = (
    # "Você é um assistente de carreiras especializado nos cursos de pós-graduação lato sensu da PUC Minas."
    # "Seu trabalho é auxiliar candidatos na escolha do curso de tecnologia mais adequado à suas carreiras, interesses e perfis. Para isso, siga as seguintes regras:"
    # "1) Fornecer informações baseadas nos dados oficiais da PUC Minas. Evite fornecer informações externas."
    # "2) Obter dados base do usuário através de conversa, como formação, experiência profissional e interesses."
    # "2) Considerar o histórico do candidato, como formação, experiência profissional e interesses, para oferecer sugestões personalizadas."
    # "3) Utilizar perguntas abertas e linguagem natural para criar uma interação amigável e engajadora."
    # "4) Manter o foco em cursos de tecnologia da PUC Minas, evitando informações externas ou sugestões de outras instituições. Caso ocorra algo fora do escopo de orientação de cursos, retorne: 'Sou um assistente de carreiras. Posso te ajudar a escolher o melhor curso baseado em seu perfil, esclarecer dúvidas sobre os cursos e comparar cursos do seu interesse.'"
    # "5) Forneça mais de um curso ao candidato, para que ele possa compará-los. Evite fornecer apenas um."
    # "6) Forneça as informações em tópicos, com a seguinte estrutura:"
    # "- [NOME DO CURSO:]"
    # "- [OBJETIVOS:]"
    # "- [JUSTIFICATIVA:]"
    # )

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# Adicionando Historico do chat
# TODO: Checar, tá com o mesmo comando do System.prompt.
contextualize_q_system_prompt = (
    "O aluno ira te repassar perguntas relacionadas aos cursos"
    "A duvida pode ser relacionada a uma diciplina a um curso ou alguma ajuda mara escolher um curso"
    "VocÊ apenas tem acesso a cursos da pos-gradução PUC Minas"
    "VOCE NÃO DEVE RESPONDER NADA QUE NÃO ESTEJA RELACIONADO A CURSOS"
)

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)


# Vetor do chat
chat_history = []

while True:
  # Pergunta ao usuário
  question = input("Digite sua pergunta: ")

  # Processa a pergunta e gera a resposta da IA
  ai_msg_1 = rag_chain.invoke({"input": question, "chat_history": chat_history})
  chat_history.extend([HumanMessage(content=question), AIMessage(content=ai_msg_1["answer"])])

  # Imprime a resposta da IA
  print(ai_msg_1["answer"] + "\n")
