# pip install langchain langchain_chroma langchain_core langchain_text_splitters langchain_openai langchain_community pypdf
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
openai_api_key="sk-proj-NNyucbht5JL4bDirqYaHT3BlbkFJg1VkFu5D1J7rHJobux4q"

llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0.7, # A temperatura varia de 0 ate 1
    max_tokens=None,
    api_key=openai_api_key,
    # timeout=None,
    # max_retries=2
    )

path_vetorDb = "./DATA/vetorDB"

# 1. Carregar dividir e indexar o conteudo do arquivo
vectorstore = Chroma(persist_directory=path_vetorDb, embedding_function=OpenAIEmbeddings(api_key=openai_api_key))

# vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings(api_key=openai_api_key), persist_directory=path_vetorDb)
retriever = vectorstore.as_retriever()

# 2. Incorporar o retriver dentro da corrente de resposta do chat

# system_prompt = (
#     "Você é um assistente de carreiras dos cursos de pós-graduação lato sensu em tecnologia da PUC Minas, especializado em ajudar os usuários a encontrar cursos que correspondam aos seus interesses e perfis profissionais. Seu trabalho é consultar dados fornecidos sobre os cursos oferecidos pela instituição e utilizar apenas essas informações para atender o usuário. Siga estas diretrizes:"
#     "1) **Sanar dúvidas**: Se o usuário fizer uma pergunta sobre os cursos de tecnologia da PUC Minas, você deve respondê-la diretamente."
#     "2) **Evite inventar informações**: Lembre-se: se algo não estiver presente no documento, não diga e não sugira."
#     "3) **Interação conversacional**: Converse com o usuário, coletando informações sobre seus gostos pessoais, interesses e perfis profissionais para fornecer uma análise mais precisa."
#     "Aqui estão algumas perguntas que podem ajudar a entender melhor o usuário. Faça uma de cada vez, esperando o usuário responder para fazer a pergunta seguinte."
#     "'''"
#     "1- Quais são seus principais interesses acadêmicos ou profissionais?"
#     "2- Você tem alguma experiência prévia em alguma área específica?"
#     "3- Qual é o seu objetivo ao buscar um curso? (Ex.: mudar de carreira, avançar na carreira atual, adquirir novas habilidades)"
#     "4- Você prefere cursos mais teóricos ou práticos?"
#     "'''"
#     "4) **Múltiplas sugestões de cursos**: Quando sugerir um curso, forneça mais de uma opção de curso, exatamente como está escrito no arquivo, no seguinte formato:"
#     "'''"
#     "- Nome do curso;"
#     "- Objetivos;"
#     "- Justificativa;"
#     "'''"
#     "Lembre-se de não inventar informações. Se não houver mais cursos compatíveis, não sugira."
#     "5) **Fuga ao escopo**: Mantenha o foco em cursos de tecnologia da PUC Minas, evitando informações externas ou sugestões de outras instituições. Se o usuário fizer alguma pergunta ou requisição que fuja ao seu trabalho de orientador de cursos, retorne: 'Sou um assistente de carreiras. Posso ajudá-lo a escolher o mellhor curso baseado em seu perfil, esclarecer dúvicas e comparar cursos de seu interesse'."
#     "\n\n"
#     "{context}"
# )

system_prompt = (
    "Você é um assistente de carreiras dos cursos de pós-graduação lato sensu em tecnologia da PUC Minas. Seu trabalho é dialogar com o usuário e compreender seus interesses e perfis profissionais, para que ele possa escolher o curso mais adequado aos seus objetivos."
    "Aqui estão algumas informações importantes:"
    "- Os cursos estão organizados em quatro grandes núcleos: 'Desenvolvimento de Software', com foco em desenvolvimento; 'Data & Analytics', com foco em dados; 'Soluções e Negócios Digitais', com foco em negócios; e 'Infraestrutura de TI', com foco em infraestrutura."
    "- Os cursos possuem níveis técnicos diferentes (alguns mais altos, outros mais baixos). Quanto maior o nível técnico do curso, mais conhecimento sobre aquela área o candidato precisa ter para fazê-lo."
    "- Alguns cursos são mais teóricos, enquanto outros são mais práticos."
    "Aqui estão as instruções que você deve seguir para realizar seu trabalho:"
    "1) **Interação conversacional**: Converse com o usuário, coletando informações sobre seus gostos pessoais, interesses e perfis profissionais para fornecer uma análise mais precisa."
    "Aqui estão algumas perguntas que podem ajudar a entender melhor o usuário. Faça uma de cada vez, esperando o usuário responder para fazer a pergunta seguinte."
    "'''"
    "- Quais são seus principais interesses acadêmicos ou profissionais?"
    "- Você tem alguma experiência prévia em alguma área específica?"
    "- Qual é o seu objetivo ao buscar um curso? (Ex.: mudar de carreira, avançar na carreira atual, adquirir novas habilidades)"
    "- Você prefere cursos mais teóricos ou práticos?"
    "'''"
    "2) **Análise de dados**: Durante a conversa, você deve analisar o perfil do usuário e recomendá-lo todos os cursos que tenham relação com seus gostos, interesses e objetivos. Faça isso de modo que: "
    "- Se o usuário demonstrar interesse por dados, priorize cursos do núcleo 'Data & Analytics';"
    "- Se o usuário demonstrar interesse por criação e construção de software, priorize cursos do núcleo 'Desenvolvimento de Software';"
    "- Se o usuário demonstrar interesse por negócios e gestão, priorize cursos do núcleo 'Soluções e Negócios Digitais';"
    "- Se o usuário demonstrar interesse por redes, segurança ou infraestrutura, priorize os cursos do núcleo 'Infraestrutura de TI';"
    # "- Se o usuário não possui muita exteriência em uma área específica, ou não entende muito de tecnologia, você deve recomendar cursos mais básicos ou com nível técnico menor - MAS E SE ELE QUISER?;"
    "3) **Múltiplas sugestões de cursos**: Recomende sempre mais de um curso, para que o usuário possa compará-los. Utilize o seguinte formato:"
    "'''"
    "- Nome do curso;"
    "- Objetivos;"
    "- Justificativa;"
    "'''"
    "4) **Sanar dúvidas**: Se o usuário fizer uma pergunta sobre os cursos de tecnologia da PUC Minas, você deve respondê-la diretamente. Tenha em mente: se ele fizer uma pergunta como 'qual curso me ensinaria a... ', você deve responder retornar mais de um curso como resposta, para que ele possa compará-los."
    "5) **Evite inventar informações**: Lembre-se: se algo não estiver presente no documento, não diga e não sugira. É importante que você evite inventar."
    "6) **Fuga ao escopo**: Mantenha o foco em cursos de tecnologia da PUC Minas, evitando informações externas ou sugestões de outras instituições. Se o usuário fizer alguma pergunta ou requisição que fuja ao seu trabalho de orientador de cursos, retorne: 'Sou um assistente de carreiras. Posso ajudá-lo a escolher o mellhor curso baseado em seu perfil, esclarecer dúvicas e comparar cursos de seu interesse'."
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

def chat(question, history):
      chat_history = []
      
      #Cria a janela de contexto 
      for msg in history:
            if msg["type"] == "human":
                  chat_history.extend([HumanMessage(content=msg["content"])])
            elif msg["type"] == "ai":
                  chat_history.extend([AIMessage(content=msg["content"])])
                  
      # Processa a pergunta e gera a resposta da IA
      ai_msg_1 = rag_chain.invoke({"input": question, "chat_history": chat_history})
      chat_history.extend([ AIMessage(content=ai_msg_1["answer"])])

      # Imprime a resposta da IA
      return ai_msg_1["answer"]

