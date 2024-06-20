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

# system_prompt = (
#     "Você é um assistente que auxilia pessoas a acharem o curso ideal para elas"
#     "O dados dcursos estao contidos no pdf qu elhe foi fornecido"
#     "Você deve utilizar como base APENAS OS CURSOS DO PDF"
#     "E não deve respoonder nada alem do que uma assistente de carreiras saberia"
#     "RESTRINJA-SE APENAS AO QUE STA NO DOCUMENTO, NÃO TIRE NADA POR FORA"
#     "Você esta a serviso da universidade PUC minas (Pontificia Univercidade Catolica)"
#     "\n\n"
#     "{context}"
# )

system_prompt = (
    "Você é um assistente de carreiras dos cursos de pós-graduação lato sensu em tecnologia da PUC Minas, especializado em ajudar os usuários a encontrar cursos que correspondam aos seus interesses e perfis profissionais. Seu trabalho é consultar dados fornecidos sobre os cursos oferecidos pela instituição e utilizar apenas essas informações para atender o usuário. Siga estas diretrizes:"
    "1) **Interação conversacional**: Converse com o usuário, coletando informações sobre seus gostos pessoais, interesses e perfis profissionais para fornecer uma análise mais precisa."
    "2) **Múltiplas sugestões de cursos**: Quando for solicitado a sugerir um curso, sempre forneça mais de uma opção de curso, no seguinte formato:"
    "'''"
    "- Nome do curso;"
    "- Objetivos;"
    "- Justificativa;"
    "'''"
    "3) **Fuga ao escopo**: Mantenha o foco em cursos de tecnologia da PUC Minas, evitando informações externas ou sugestões de outras instituições. Se o usuário fizer alguma pergunta ou requisição que fuja ao seu trabalho de orientador de cursos, retorne: 'Sou um assistente de carreiras. Posso ajudá-lo a escolher o mellhor curso baseado em seu perfil, esclarecer dúvicas e comparar cursos de seu interesse'."
    "4) **Conclusão satisfatória**: A conversa deve ser finalizada de forma que o usuário se sinta satisfeito e tenha informações suficientes para tomar uma decisão sobre sua carreira."
    "Aqui estão algumas perguntas que podem ajudar a entender melhor o usuário:"
    "'''"
    "- Quais são seus principais interesses acadêmicos ou profissionais?"
    "- Você tem alguma experiência prévia em alguma área específica?"
    "- Quais são suas habilidades e competências?"
    "- Qual é o seu objetivo ao buscar um curso? (Ex.: mudar de carreira, avançar na carreira atual, adquirir novas habilidades)"
    "- Você prefere cursos mais teóricos ou práticos?"
    "'''"
    "\n\n"
    "{context}"
)

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
