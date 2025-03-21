# Assistente de carreiras

- [Motivação](#Motivação)
- [Ferramentas](#Ferramentas)
- [Partições do Código](#Partições-do-código)
- [Run](#Run)

## Motivação

A motivação deste código foi a necessidade de um assistente para guiar os alunos na escolha de curso/carreiraa, porém o ChatGPT não tinha acesso aos cursos da PUC Minas. Então tivemos que descobrir uma maneira de passar essas informações para o LLM.

## Ferramentas

### Langchain

É um framework que auxilia e trabalha em conjunto com o LLM, permitindo que ele receba documentos e opere com uma base de dados mais específica — neste caso, os projetos pedagógicos dos cursos de pós-graduação da PUC Minas.

### OpenAI

###### É a biblioteca do ChatGPT que nos ajuda com:
- Chat: Permite criar assistentes conversacionais, responder perguntas, gerar textos, resumos, análises, etc.
- Embeddings: Transforma textos em vetores numéricos, úteis para busca semântica, RAG (Retrieval-Augmented Generation), clustering, etc.
- DALL·E: Gera imagens a partir de descrições em linguagem natural.
- Whisper: Converte áudios em texto (transcrição automática).
- Text-to-Speech: Converte textos em áudios com vozes realistas.
- Assistants API: Cria agentes com ferramentas (código, busca, arquivos, etc), memória e objetivos.
- Fine-tuning: Permite ajustar modelos para casos específicos com dados próprios.
- Moderação: Detecta conteúdo impróprio ou sensível em textos.

## Partições do Código

### Processamento dos documentos

Inicialmente, para que possamos trabalhar com os documentos, é necessário convertê-los para um formato que a IA compreenda e possa navegar internamente. Por isso, é preciso realizar a importação e o fracionamento (split) dos documentos.

### Elaboração do prompt de configuração

Para que a IA compreenda o contexto no qual está inserida, é fundamental fornecer um prompt de configuração adequado, que delimite o universo de atuação e suas regras.

### Parte do contexto

...

### Chat

Essa etapa envolve apenas o envio do histórico de mensagens para a IA, o recebimento das perguntas e a geração das respostas.

## Run

Para instalar todas as dependências necessárias, é preciso aguardar a conclusão do comando abaixo.

```bash
pip install langchain langchain_chroma langchain_core langchain_text_splitters langchain_openai langchain_community
```

> [!important]
> É recomendado que tenha a chave da api (OPENAI_API_KEY) COLOCADA NAS VARIAVEIS DE AMBIENTE (.env) para que não seja exposta.
> porem caso não seja possivel altere a linha de [configuração](https://github.com/WebTech-PUC-Minas/assistente-carreiras-gpt/blob/main/lang_chain_cursos.py#L17 "Click para ir para a linha do codigo") da OPENAI
