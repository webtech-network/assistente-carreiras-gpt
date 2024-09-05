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

# Split no texto e criação da vectorstore com os tokens

text_splitter: RecursiveCharacterTextSplitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
"""
Cria uma instância de RecursiveCharacterTextSplitter, um utilitário para dividir textos longos em pedaços menores.
- chunk_size=1000: Define o tamanho máximo de cada pedaço (chunk) de texto em caracteres.
- chunk_overlap=200: Define a quantidade de sobreposição (em caracteres) entre os chunks consecutivos.
  Isso é útil para garantir que o contexto não se perca entre as divisões de texto.
"""

splits: list[Dict[str, Any]] = text_splitter.split_documents(docs)
"""
Usa o método split_documents() da instância text_splitter para dividir o conteúdo de 'docs' em pedaços menores.
- docs: A variável que contém os documentos carregados do PDF, que são inicialmente um grande bloco de texto.
- splits: Uma lista de pedaços de texto menores, cada um de tamanho aproximadamente definido por chunk_size, 
  com uma sobreposição de chunk_overlap para manter o contexto entre os pedaços.
"""

vectorstore: Chroma = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings(api_key=openai_api_key))
"""
Cria um vetor de armazenamento (vectorstore) usando a biblioteca Chroma para armazenar os embeddings dos textos divididos.
- Chroma.from_documents: Um método estático que cria um armazenamento de vetores a partir de uma lista de documentos.
  - documents=splits: Os pedaços de texto divididos que serão convertidos em embeddings.
  - embedding=OpenAIEmbeddings(api_key=openai_api_key): Instância de OpenAIEmbeddings usada para converter cada pedaço de texto em um vetor numérico (embedding).
    - OpenAIEmbeddings: Classe que usa a API da OpenAI para gerar embeddings. Precisa de uma chave de API (api_key) para autenticação.
- vectorstore: Um objeto que armazena embeddings para todos os pedaços de texto fornecidos, permitindo operações de recuperação e busca baseadas em similaridade semântica.
"""

retriever = vectorstore.as_retriever()
"""
Converte o vectorstore em um retriever, que é uma interface para buscar textos semelhantes.
- vectorstore.as_retriever(): Método que cria um objeto retriever a partir do vectorstore. O retriever é usado para consultas de recuperação de documentos.
- retriever: Um objeto que permite buscar pedaços de texto que são semanticamente similares a uma query fornecida. Isso é útil em aplicações como chatbots, sistemas de recomendação e motores de busca de documentos.
"""


# 2. Incorporar o retriver dentro da corrente de resposta do chat
#Formato que o framework passa as mensagens do sistema e usuário para chain
#System Prompt
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

#TODO: entender essa parte
question_answer_chain: Runnable[dict[str, Any], Any] = create_stuff_documents_chain(llm, prompt)
rag_chain: object = create_retrieval_chain(retriever, question_answer_chain)

# Adicionando Historico do chat
contextualize_q_system_prompt: str = (
    "O aluno ira te repassar perguntas relacionadas aos cursos"
    "A duvida pode ser relacionada a uma diciplina a um curso ou alguma ajuda mara escolher um curso"
    "VocÊ apenas tem acesso a cursos da pos-gradução PUC Minas"
    "VOCE NÃO DEVE RESPONDER NADA QUE NÃO ESTEJA RELACIONADO A CURSOS"
)

# Não entendi a redundância nos prompts
contextualize_q_prompt: ChatPromptTemplate = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# Qual a função dessas chains na arquitetura?
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

# Prompt de pergunta e resposta
# Ele muda o prompt denovo, adiciona ele
qa_prompt: ChatPromptTemplate = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# Outra rag chain
question_answer_chain: Runnable[dict[str, Any], Any] = create_stuff_documents_chain(llm, qa_prompt)
rag_chain: object  = create_retrieval_chain(history_aware_retriever, question_answer_chain)

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

#Testes LangSmith

