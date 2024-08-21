#%% [markdown]
## Otimização
- Sim, o LangSmith pode ser usado para otimizar uma pipeline de RAG (Retrieval-Augmented Generation) e melhorar a acurácia das respostas do modelo assistente. Aqui estão algumas estratégias usando LangSmith e outras abordagens complementares:

### 1. **Otimização com LangSmith**:
- **Ajuste de prompts**: Use LangSmith para criar e testar diferentes prompts e fluxos de conversa, ajustando os parâmetros para maximizar a relevância e a precisão das respostas.
- **Monitoramento e feedback**: LangSmith pode coletar dados sobre as interações e fornecer insights sobre onde o modelo pode estar falhando, permitindo ajustes iterativos.
- **Avaliação de performance**: Utilize o LangSmith para comparar diferentes versões da pipeline RAG e avaliar qual delas oferece melhores resultados em termos de acurácia e relevância.

### 2. **Estratégias Complementares**:
- **Reforço por feedback humano**: Incorporar técnicas de aprendizado por reforço (Reinforcement Learning) onde o feedback humano é utilizado para refinar o modelo, melhorando a qualidade das respostas ao longo do tempo.
- **Afinamento fino (fine-tuning)**: Realizar o fine-tuning do modelo com dados específicos do domínio de aplicação para melhorar a acurácia nas respostas.
- **Melhorias na recuperação de documentos**: Melhorar o módulo de recuperação (retrieval) para garantir que as informações mais relevantes sejam apresentadas ao modelo para geração, o que pode incluir a utilização de técnicas avançadas de indexação e recuperação semântica.
#%%
### 1. **Ajuste de Prompts**:
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langsmith import LangSmith

# Conectando com o LangSmith para rastrear interações
langsmith = LangSmith(api_key="your_langsmith_api_key")

# Template de prompt personalizado
template = """
You are a helpful assistant. Based on the following context, answer the question concisely.

Context: {context}
Question: {question}
Answer:
"""

prompt = PromptTemplate(template=template)

# Configurando a cadeia de recuperação aumentada
chain = ConversationalRetrievalChain.from_chain_type(
    retriever=retriever,  # seu retriever personalizado
    qa_chain_type="stuff",
    qa_prompt=prompt
)

# Rastreando a interação
response = langsmith.track(chain, question="What is LangSmith?", context="LangSmith is a tool...")

print(response)
#%%
### 2. **Monitoramento e Feedback**:
from langsmith import LangSmith, FeedbackCollector

# Inicializar LangSmith e o coletor de feedback
langsmith = LangSmith(api_key="your_langsmith_api_key")
feedback_collector = FeedbackCollector(langsmith)

# Exemplo de resposta e feedback
response = chain.run(question="What is the capital of France?")
feedback = feedback_collector.collect_feedback(response=response, correct_answer="Paris")

# Armazenando o feedback para análise posterior
langsmith.store_feedback(feedback)
#%%
### 3. **Comparação de Versões da Pipeline**:
# Definindo diferentes pipelines para comparação
chain_v1 = ConversationalRetrievalChain.from_chain_type(
    retriever=retriever_v1,
    qa_chain_type="stuff",
    qa_prompt=prompt_v1
)

chain_v2 = ConversationalRetrievalChain.from_chain_type(
    retriever=retriever_v2,
    qa_chain_type="map_reduce",
    qa_prompt=prompt_v2
)

# Executando ambas as versões para a mesma consulta
response_v1 = langsmith.track(chain_v1, question="What is LangSmith?", context="LangSmith is a tool...")
response_v2 = langsmith.track(chain_v2, question="What is LangSmith?", context="LangSmith is a tool...")

# Comparando resultados
comparison = langsmith.compare_responses(response_v1, response_v2)

print(comparison)
#%%
### 4. **Melhoria da Recuperação de Documentos**:
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings

# Usando FAISS como backend de recuperação
vectorstore = FAISS.from_documents(documents, OpenAIEmbeddings())

# Integrando com o LangSmith
retriever = vectorstore.as_retriever()

# Otimizando a recuperação com ajustes finos
retriever.search_kwargs['k'] = 5  # Ajuste do número de documentos recuperados

# Conectando com LangSmith para rastrear interações
chain = ConversationalRetrievalChain.from_chain_type(
    retriever=retriever,
    qa_chain_type="stuff",
    qa_prompt=prompt
)

response = langsmith.track(chain, question="What is LangSmith?", context="LangSmith is a tool...")
print(response)

#%%
### Resumo das Ações:
# 1. **Ajuste de prompts**: Crie prompts personalizados que se ajustem ao contexto específico.
# 2. **Monitoramento e Feedback**: Use LangSmith para monitorar e coletar feedback sobre as respostas.
# 3. **Comparação de Pipelines**: Teste diferentes versões da pipeline e use o LangSmith para compará-las.
# 4. **Melhoria da Recuperação**: Otimize o módulo de recuperação utilizando técnicas avançadas de indexação e ajuste de parâmetros.

# Esses passos irão ajudar a melhorar tanto a acurácia quanto a relevância das respostas geradas pela sua pipeline RAG.