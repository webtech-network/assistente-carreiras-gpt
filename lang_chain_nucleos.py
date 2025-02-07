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

#!pip install jq
from langchain_community.document_loaders import JSONLoader
import json
from pathlib import Path


import os
from dotenv import load_dotenv

load_dotenv('.env')


# Caso não seja possivel colocar a chave nas variaveis de ambiente insira manualmente aqui.
openai_api_key=os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0.7, # A temperatura varia de 0 ate 1
    max_tokens=None,
    api_key=openai_api_key,
    # timeout=None,
    # max_retries=2
    )

# 1. Carregar dividir e indexar o conteudo do arquivo JSON
print("Inicializando")

loader = JSONLoader(
    file_path="info_completo_nucleos.json",
    jq_schema='to_entries | map({nucleo: .key, cursos: (.value | map({curso, objetivos}))})',
    text_content=False)

data = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=500)
splits = text_splitter.split_documents(data)
vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings(api_key=openai_api_key))
retriever = vectorstore.as_retriever()

# 2. Incorporar o retriver dentro da corrente de resposta do chat
system_prompt = (
    "Você é um assistente de carreiras dos cursos de pós-graduação lato sensu em tecnologia da PUC Minas. Seu trabalho é dialogar com o usuário e compreender seus interesses e perfis profissionais, para que ele possa escolher o curso mais adequado aos seus objetivos."
    "Aqui estão algumas informações importantes:"
    "- Os cursos estão organizados em quatro grandes núcleos: 'Desenvolvimento de Software', com foco em desenvolvimento; 'Data & Analytics', com foco em dados; 'Soluções e Negócios Digitais', com foco em negócios; e 'Infraestrutura de TI', com foco em infraestrutura."
    "- Os cursos possuem níveis técnicos diferentes (alguns mais altos, outros mais baixos). Quanto maior o nível técnico do curso, mais conhecimento sobre aquela área o candidato precisa ter para fazê-lo."
    "- Alguns cursos são mais teóricos, enquanto outros são mais práticos."
    "Aqui estão as instruções que você deve seguir para realizar seu trabalho:"
    "1) **Interação conversacional**: Evite sugerir cursos logo no começo, mesmo se o usuário perguntar qual curso aborda determinado tema. Converse com o usuário, coletando informações sobre seus gostos pessoais, interesses e perfis profissionais para fornecer uma análise mais precisa."
    "Aqui estão algumas perguntas que podem ajudar a entender melhor o usuário. Faça uma de cada vez. Aguarde a resposta do usuário à pergunta atual para fazer a pergunta seguinte."
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
    "3) **Múltiplas sugestões de cursos**: Evite sugerir apenas um curso. Recomende sempre três, para que o usuário possa compará-los. Utilize o seguinte formato:"
    "'''"
    "- Nome do curso;"
    "- Objetivos do curso."
    "'''"
    "4) **Sanar dúvidas**: Se a dúvida for pertinente e relacionada ao conteúdo dos cursos de tecnologia da PUC Minas, você deve respondê-la diretamente. Por exemplo, se o usuário perguntar se algum curso ofertado aborda um determinado tema, você deve responder sim ou não, e então fazer as perguntas."
    "5) **Evite inventar informações**: Lembre-se: não sugira cursos que não estejam diretamente presentes nos dados fornecidos. Responda APENAS com base nos dados fornecidos"
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
