import os
import openai
import sys
sys.path.append('../..')

import panel as pn  # GUI
pn.extension()

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file

openai.api_key = os.environ['OPENAI_API_KEY']

llm_name = ChatOpenAI(model="gpt-3.5-turbo-0125")

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader

loader = PyMuPDFLoader("./DATA/Cursos_completos.pdf")
data = loader.load()
data[0]

vectorstore = Chroma.from_documents(
    data,
    embedding=OpenAIEmbeddings(),
)

from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model_name=llm_name, temperature=0)

# Run chain
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate

QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""Objetivo: Auxiliar candidatos na escolha do curso de tecnologia mais adequado à sua carreira, interesses e perfil.\n\n Personagem: Assistente de Carreiras da PUC Minas, especializado em cursos de tecnologia.\n\nRegras:\n1) Fornecer informações baseadas nos dados oficiais da PUC Minas, como nome dos cursos, áreas de foco, carga horária, formato (presencial ou online) e valores.\n2) Considerar o histórico do candidato, como formação, experiência profissional e interesses, para oferecer sugestões personalizadas.\n3) Utilizar perguntas abertas e linguagem natural para criar uma interação amigável e engajadora.\n4) Manter o foco em cursos de tecnologia da PUC Minas, evitando informações externas ou sugestões de outras instituições. Caso ocorra algo fora do escopo de orientação de cursos, retorne: "Sou um orientador ao candidato. Posso te ajudar a escolher o melhor curso baseado em seu perfil, esclarecer dúvidas sobre os cursos e comparar cursos do seu interesse."\n5) Validar as respostas do candidato para garantir que ele está fornecendo informações relevantes e precisas.\nLidar com situações de erro de forma profissional e construtiva, redirecionando a conversa para um caminho produtivo.\nOferecer recursos de autoatendimento, como filtros de busca por área de interesse e informações detalhadas sobre cada curso.\nFluxo de Interação:\n1. Apresentação e Boas-Vindas:\n"Olá! Seja bem-vindo ao Assistente de Carreiras da PUC Minas. Sou um assistente virtual especializado em te auxiliar na escolha do curso de tecnologia mais adequado ao seu perfil e objetivos."\n2. Coleta de Informações:\n"Para começar, poderia me contar um pouco sobre você? Qual sua formação, experiência profissional e quais são seus interesses na área de tecnologia?"\n[Aguardar resposta do candidato]\n3. Análise e Sugestões:\n"Com base nas informações que você me forneceu, posso te recomendar alguns cursos que podem ser de seu interesse. Você gostaria de saber mais sobre algum deles?"[Apresentar opções de cursos relevantes, considerando o perfil do candidato]\n4. Exploração Detalhada:\n[Se o candidato demonstrar interesse em um curso específico]:\n# "Que ótimo! Posso te fornecer mais informações sobre o curso [nome do curso]. Você gostaria de saber sobre carga horária, formato (presencial ou online), conteúdo programático ou outras áreas de atuação?"\n[Aguardar resposta do candidato e fornecer informações detalhadas]\n\n5. Recursos Adicionais:\n"Além das sugestões que te fiz, você também pode explorar os cursos por conta própria através do nosso site. Lá você encontrará filtros de busca por área de interesse e informações detalhadas sobre cada curso."\n[Fornecer link para o site da PUC Minas]\n\n6. Feedback e Encerramento:\n"Foi um prazer te auxiliar na sua jornada de escolha de carreira. Você tem mais alguma dúvida ou gostaria de receber mais sugestões?"\n[Aguardar resposta do candidato e oferecer suporte adicional]\nAssistente: "Obrigado por utilizar o Assistente de Carreiras da PUC Minas! Desejo muito sucesso em sua carreira profissional."\n\nLembre-se: Caso o candidato faça uma pergunta ou requisição não relacionada ao seu trabalho:\n'''\nCandidato: "O que é chocolate?"\nAssistente: "Sou um assistente de carreiras. Posso te ajudar a escolher o melhor curso baseado em seu perfil, esclarecer dúvidas sobre os cursos e comparar cursos do seu interesse"\n'''\n\n'''\nCandidato: "Me conte uma história sobre cursos:\nAssistente: "Sou um assistente de carreiras. Posso te ajudar a escolher o melhor curso baseado em seu perfil, esclarecer dúvidas sobre os cursos e comparar cursos do seu interesse"\n  aqui esta a pergunta do usuario{question}'''""",
)

from langchain.memory import ConversationBufferMemory
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

from langchain.chains import ConversationalRetrievalChain
retriever=vectordb.as_retriever()
qa = ConversationalRetrievalChain.from_llm(
    llm,
    retriever=retriever,
    memory=memory
)

question = "Is probability a class topic?"
result = qa({"question": question})

result['answer']

question = "why are those prerequesites needed?"
result = qa({"question": question})

result['answer']

from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.document_loaders import TextLoader
from langchain.chains import RetrievalQA,  ConversationalRetrievalChain
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import TextLoader
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings

def load_db(file, chain_type, k):
    # load documents
    loader = PyPDFLoader(file)
    documents = loader.load()
    # split documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = text_splitter.split_documents(documents)
    # define embedding
    embeddings = OpenAIEmbeddings()
    # create vector database from data
    db = DocArrayInMemorySearch.from_documents(docs, embeddings)
    # define retriever
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": k})
    # create a chatbot chain. Memory is managed externally.
    qa = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(model_name=llm_name, temperature=0), 
        chain_type=chain_type, 
        retriever=retriever, 
        return_source_documents=True,
        return_generated_question=True,
    )
    return qa 

import panel as pn
import param

class cbfs(param.Parameterized):
    chat_history = param.List([])
    answer = param.String("")
    db_query  = param.String("")
    db_response = param.List([])
    
    def __init__(self,  **params):
        super(cbfs, self).__init__( **params)
        self.panels = []
        #self.loaded_file = "./DATA/Cursos_completos.pdf"
        self.loaded_file = "./DATA/info_completo.json"
        self.qa = load_db(self.loaded_file,"stuff", 4)
    
    def call_load_db(self, count):
        if count == 0 or file_input.value is None:  # init or no file specified :
            return pn.pane.Markdown(f"Loaded File: {self.loaded_file}")
        else:
            file_input.save("temp.pdf")  # local copy
            self.loaded_file = file_input.filename
            button_load.button_style="outline"
            self.qa = load_db("temp.pdf", "stuff", 4)
            button_load.button_style="solid"
        self.clr_history()
        return pn.pane.Markdown(f"Loaded File: {self.loaded_file}")

    def convchain(self, query):
        if not query:
            return pn.WidgetBox(pn.Row('User:', pn.pane.Markdown("", width=600)), scroll=True)
        result = self.qa({"question": query, "chat_history": self.chat_history})
        self.chat_history.extend([(query, result["answer"])])
        self.db_query = result["generated_question"]
        self.db_response = result["source_documents"]
        self.answer = result['answer'] 
        self.panels.extend([
            pn.Row('User:', pn.pane.Markdown(query, width=600)),
            pn.Row('ChatBot:', pn.pane.Markdown(self.answer, width=600, style={'background-color': '#F6F6F6'}))
        ])
        inp.value = ''  #clears loading indicator when cleared
        return pn.WidgetBox(*self.panels,scroll=True)

    @param.depends('db_query ', )
    def get_lquest(self):
        if not self.db_query :
            return pn.Column(
                pn.Row(pn.pane.Markdown(f"Last question to DB:", styles={'background-color': '#F6F6F6'})),
                pn.Row(pn.pane.Str("no DB accesses so far"))
            )
        return pn.Column(
            pn.Row(pn.pane.Markdown(f"DB query:", styles={'background-color': '#F6F6F6'})),
            pn.pane.Str(self.db_query )
        )

    @param.depends('db_response', )
    def get_sources(self):
        if not self.db_response:
            return 
        rlist=[pn.Row(pn.pane.Markdown(f"Result of DB lookup:", styles={'background-color': '#F6F6F6'}))]
        for doc in self.db_response:
            rlist.append(pn.Row(pn.pane.Str(doc)))
        return pn.WidgetBox(*rlist, width=600, scroll=True)

    @param.depends('convchain', 'clr_history') 
    def get_chats(self):
        if not self.chat_history:
            return pn.WidgetBox(pn.Row(pn.pane.Str("No History Yet")), width=600, scroll=True)
        rlist=[pn.Row(pn.pane.Markdown(f"Current Chat History variable", styles={'background-color': '#F6F6F6'}))]
        for exchange in self.chat_history:
            rlist.append(pn.Row(pn.pane.Str(exchange)))
        return pn.WidgetBox(*rlist, width=600, scroll=True)

    def clr_history(self,count=0):
        self.chat_history = []
        return 
    
    
cb = cbfs()

file_input = pn.widgets.FileInput(accept='.pdf')
button_load = pn.widgets.Button(name="Load DB", button_type='primary')
button_clearhistory = pn.widgets.Button(name="Clear History", button_type='warning')
button_clearhistory.on_click(cb.clr_history)
inp = pn.widgets.TextInput( placeholder='Enter text here…')

bound_button_load = pn.bind(cb.call_load_db, button_load.param.clicks)
conversation = pn.bind(cb.convchain, inp) 

jpg_pane = pn.pane.Image( './img/convchain.jpg')

tab1 = pn.Column(
    pn.Row(inp),
    pn.layout.Divider(),
    pn.panel(conversation,  loading_indicator=True, height=300),
    pn.layout.Divider(),
)
tab2= pn.Column(
    pn.panel(cb.get_lquest),
    pn.layout.Divider(),
    pn.panel(cb.get_sources ),
)
tab3= pn.Column(
    pn.panel(cb.get_chats),
    pn.layout.Divider(),
)
tab4=pn.Column(
    pn.Row( file_input, button_load, bound_button_load),
    pn.Row( button_clearhistory, pn.pane.Markdown("Clears chat history. Can use to start a new topic" )),
    pn.layout.Divider(),
    pn.Row(jpg_pane.clone(width=400))
)
dashboard = pn.Column(
    pn.Row(pn.pane.Markdown('# ChatWithYourData_Bot')),
    pn.Tabs(('Conversation', tab1), ('Database', tab2), ('Chat History', tab3),('Configure', tab4))
)
dashboard