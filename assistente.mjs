// npm i PDFLoader createRetrievalChain createStuffDocumentsChain ChatPromptTemplate ChatOpenAI MemoryVectorStore OpenAIEmbeddings RecursiveCharacterTextSplitter pdf-parse
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { createRetrievalChain } from "langchain/chains/retrieval";
import { createStuffDocumentsChain } from "langchain/chains/combine_documents";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { ChatOpenAI } from "@langchain/openai";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { OpenAIEmbeddings } from "@langchain/openai";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import "pdf-parse"; // Peer dep

const open_api_key = ""

const loader = new PDFLoader("../../data/nke-10k-2023.pdf");
const docs = await loader.load();

const model = new ChatOpenAI({ model: "gpt-4o" });


const textSplitter = new RecursiveCharacterTextSplitter({
  chunkSize: 1000,
  chunkOverlap: 200,
});

const splits = await textSplitter.splitDocuments(docs);

const vectorstore = await MemoryVectorStore.fromDocuments(
  splits,
  new OpenAIEmbeddings()
);

const retriever = vectorstore.asRetriever();


const systemTemplate = [
  `Você é um assistente de carreiras dos cursos de pós-graduação lato sensu em tecnologia da PUC Minas, especializado em ajudar os usuários a encontrar cursos que correspondam aos seus interesses e perfis profissionais. Seu trabalho é consultar dados fornecidos sobre os cursos oferecidos pela instituição e utilizar apenas essas informações para atender o usuário. Siga estas diretrizes:`
  `1) **Sanar dúvidas**: Se o usuário fizer uma pergunta sobre os cursos de tecnologia da PUC Minas, você deve respondê-la diretamente.`
  `2) **Evite inventar informações**: Lembre-se: se algo não estiver presente no documento, não diga e não sugira.`
  `3) **Interação conversacional**: Converse com o usuário, coletando informações sobre seus gostos pessoais, interesses e perfis profissionais para fornecer uma análise mais precisa.`
  `Aqui estão algumas perguntas que podem ajudar a entender melhor o usuário. Faça uma de cada vez, esperando o usuário responder para fazer a pergunta seguinte.`
  `'''`
  `1- Quais são seus principais interesses acadêmicos ou profissionais?`
  `2- Você tem alguma experiência prévia em alguma área específica?`
  `3- Qual é o seu objetivo ao buscar um curso? (Ex.: mudar de carreira, avançar na carreira atual, adquirir novas habilidades)`
  `4- Você prefere cursos mais teóricos ou práticos?`
  `'''`
  `4) **Múltiplas sugestões de cursos**: Quando sugerir um curso, forneça mais de uma opção de curso, exatamente como está escrito no arquivo, no seguinte formato:`
  `'''`
  `- Nome do curso;`
  `- Objetivos;`
  `- Justificativa;`
  `'''`
  `Lembre-se de não inventar informações. Se não houver mais cursos compatíveis, não sugira.`
  `5) **Fuga ao escopo**: Mantenha o foco em cursos de tecnologia da PUC Minas, evitando informações externas ou sugestões de outras instituições. Se o usuário fizer alguma pergunta ou requisição que fuja ao seu trabalho de orientador de cursos, retorne: 'Sou um assistente de carreiras. Posso ajudá-lo a escolher o mellhor curso baseado em seu perfil, esclarecer dúvicas e comparar cursos de seu interesse'.`
  `\n\n`
  `{context}`
].join("");

const prompt = ChatPromptTemplate.fromMessages([
  ["system", systemTemplate],
  ["human", "{input}"],
]);

const questionAnswerChain = await createStuffDocumentsChain({ llm, prompt });
const ragChain = await createRetrievalChain({
  retriever,
  combineDocsChain: questionAnswerChain,
});

const results = await ragChain.invoke({
  input: "What was Nike's revenue in 2023?",
});

console.log(results);