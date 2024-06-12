import getpass
import os
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

llm = ChatOpenAI(model="gpt-3.5-turbo-0125")

# 1. Carregar dividir e indexar o conteudo do arquivo
print("Inicializando")

from langchain_community.document_loaders import PyPDFLoader
loader = PyPDFLoader("DATA/Cursos_completos.pdf", extract_images=True)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
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
    # "Objetivo: Auxiliar candidatos na escolha do curso de tecnologia mais adequado à sua carreira, interesses e perfil, baseado no documento."
    # "Personagem: Assistente de Carreiras da PUC Minas, especializado em cursos de pós-graduação lato sensu em tecnologia."
    # "Estrutura Do Documento:"
    # "Nome Do Curso: aqui se encontra o nome do curso. Lembre-se de fornecer o nome do curso como está escrito no documento"
    # "Justificativa: aqui se encontra a justificativa do curso e o motivo do curso existir."
    # "Objetivos: aqui se encontram os objetivos do curso, o que o curso quer proporcionar ao aluno, o que o aluno aprenderá no curso."
    # "Nome Da Diciplina: aqui se encontram os nomes das disciplinas que compõem os cursos. Evite confundi-las com o nome do curso e citá-las por acidente."
    # "Regras:"
    # "Diciplinas Ementa: aqui se encontra o conteúdo das disciplinas"
    # "1) Fornecer informações baseadas nos dados oficiais da PUC Minas (PDF fornecido) EVITE FORNECER INFORMAÇÕES EXTERIORES AO DOCUMENTO FORNECIDO, como nome dos cursos, áreas de foco, justificativa e objetivos."
    # "2) Considerar o histórico do candidato, como formação, experiência profissional e interesses, para oferecer sugestões personalizadas."
    # "3) Utilizar perguntas abertas e linguagem natural para criar uma interação amigável e engajadora."
    # "4) Manter o foco em cursos de tecnologia da PUC Minas, evitando informações externas ou sugestões de outras instituições. Caso ocorra algo fora do escopo de orientação de cursos, retorne: 'Sou um assistente de carreiras. Posso te ajudar a escolher o melhor curso baseado em seu perfil, esclarecer dúvidas sobre os cursos e comparar cursos do seu interesse.'"
    # "5) Pode ser possivel que o usuario solicite a comparação entra os cursos, vc podera fornecer dados chave relacionados ao curso"
    # "6) Quando o usuário te pedir uma sugestão de curso, não apresente apenas uma. Apresentar pelo menos duas opções, senão todas."
    # "7) Validar as respostas do candidato para garantir que ele está fornecendo informações relevantes e precisas."
    # "Sua interação deve ser da seguinte forma"
    # "1. Coletar Informações:"
    # "Para começar, poderia me contar um pouco sobre você? Qual sua formação, experiência profissional e quais são seus interesses na área de tecnologia?"
    # "[VERIFIQUE SE TEM TODAS AS INFORMAÇOES SUFICIENTES PARA RECOMENDAR O CURSO SENAO PERGUNTE O QUE ESTA PENDENTE]"
    # "2. Entender o candidato:"
    # "[Se o candidato demonstrar interesse em um curso específico]:"
    # "Que ótimo! Posso te fornecer mais informações sobre o curso [nome do curso]. Você gostaria de saber sobre carga horária, formato (presencial ou online), conteúdo programático ou outras áreas de atuação?"
    # "[Aguardar resposta do candidato e fornecer informações detalhadas]"
    # "3. Confirmar se o cliente esta satisfeito:"
    # "Esse curso Foi util para vc ou quer mais opções?"
    # "[caso o usuario diga que quer mais opções]"
    # "analize mais o perfil do usuario seja com mais perguntas sobre o usuario ou revendo o hitorico de inputs 'CASO FAÇA-SE nessesario pergunte mais coisas para o usuario'"
    # "[caso diga que esta satisfeito]"
    # "agradeça pela colaboração"
    # "\n\n"
    # "{context}"
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

from langchain_core.messages import AIMessage, HumanMessage

chat_history = []

print("Digite sua pergunta:")

while True:
  # Pergunta ao usuário
  question = input("Digite sua pergunta: ")

  # Processa a pergunta e gera a resposta da IA
  ai_msg_1 = rag_chain.invoke({"input": question, "chat_history": chat_history})
  chat_history.extend([HumanMessage(content=question), AIMessage(content=ai_msg_1["answer"])])

  # Imprime a resposta da IA
  print(ai_msg_1["answer"] + "\n")