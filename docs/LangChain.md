# LangChain

- Documentação explicando o funcionamento das partes do LangChain

## Bibliotecas 
Claro! Vamos dividir e explicar cada biblioteca por sua utilidade.

### Bibliotecas para Cadeias de Documentos e Recuperação

1. **`langchain.chains.combine_documents.create_stuff_documents_chain`**
   - **Função:** Cria uma cadeia para combinar documentos em um único documento ou resposta.
   - **Utilidade:** Útil para combinar múltiplos documentos em uma única resposta coesa.

2. **`langchain.chains.create_retrieval_chain`**
   - **Função:** Cria uma cadeia para recuperar documentos relevantes com base em uma consulta.
   - **Utilidade:** Útil para buscar informações relevantes em um conjunto de documentos com base em uma consulta específica.

3. **`langchain.chains.create_history_aware_retriever`**
   - **Função:** Cria um recuperador de documentos que leva em consideração o histórico de interações.
   - **Utilidade:** Útil para melhorar a precisão da recuperação de informações considerando o contexto histórico das consultas anteriores.

### Biblioteca para Armazenamento e Recuperação

4. **`langchain_chroma.Chroma`**
   - **Função:** Interface para a integração com a base de dados Chroma.
   - **Utilidade:** Útil para armazenar e recuperar embeddings e outros dados relacionados a documentos.

### Bibliotecas para Modelos e Prompts

5. **`langchain_core.prompts.ChatPromptTemplate`**
   - **Função:** Cria templates para prompts de chat.
   - **Utilidade:** Útil para definir a estrutura de prompts que serão usados em interações de chat.

6. **`langchain_core.prompts.MessagesPlaceholder`**
   - **Função:** Placeholder para mensagens em um prompt de chat.
   - **Utilidade:** Facilita a inclusão de mensagens dinâmicas em prompts de chat.

7. **`langchain_core.messages.AIMessage` e `langchain_core.messages.HumanMessage`**
   - **Função:** Define tipos de mensagens para interações de chat (mensagens de IA e mensagens de humanos).
   - **Utilidade:** Utilizado para estruturar e diferenciar mensagens de IA e de usuários em interações de chat.

### Biblioteca para Divisão de Texto

8. **`langchain_text_splitters.RecursiveCharacterTextSplitter`**
   - **Função:** Divide textos longos em partes menores, recursivamente, por caracteres.
   - **Utilidade:** Útil para processar e analisar textos longos de forma mais eficiente, dividindo-os em segmentos menores.

### Bibliotecas para Embeddings e Modelos de Linguagem

9. **`langchain_openai.OpenAIEmbeddings`**
   - **Função:** Interface para obter embeddings de textos usando modelos da OpenAI.
   - **Utilidade:** Útil para converter textos em vetores de embeddings que podem ser usados para busca e análise.

10. **`langchain_openai.ChatOpenAI`**
    - **Função:** Interface para usar modelos de chat da OpenAI.
    - **Utilidade:** Útil para realizar interações de chat com modelos de linguagem da OpenAI.

### Biblioteca para Carregamento de Documentos

11. **`langchain_community.document_loaders.PyPDFLoader`**
    - **Função:** Carrega e processa documentos PDF.
    - **Utilidade:** Útil para extrair texto e informações de arquivos PDF para análise e processamento adicional.

Essas bibliotecas e funções são combinadas para criar fluxos de trabalho de processamento e recuperação de documentos de forma eficiente e estruturada.