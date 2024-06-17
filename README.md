# Assistente de carreiras

- [Motivação](#Motivação)
- [Ferramentas](#Ferramentas)
- [Partições do Código](#Partições-do-código)
- [Run](#Run)

## Motivação

A motivação deste código foi que precisavaomos de um assistente porem o chatGPT não tinha acesso as especificidades dos cursos da PUC minas então tinhamos que arranjar alguma forma de passar essas informações para a IA

## Ferramentas

### Langchain

É uma ferramenuqe que auxilia e trabalha em conjunto com a ia para que esta possa receber documentos e trabalhar com a uma base de dados mais especifica nesse casso foram os projetos pedagogicos dos cursos de pos-graduação da PUC minas

### OpenAI

É a biblioteca do chatgpt que nos ajuda com os embadings e tambéma efitivamente fazer o chat bot

## Partições do Código

### Processamento dos documentos

Inicialmente para que possomos trabalhar com o documento precisamos passalo de uma forma que a IA entenda e que consiga tranzitar por dentro deles, por isso temos que fazer o import e o split dos documentos

### Elaboração do prompt de configuração

Para que a IA possa entender sobre o universo onde ela esta iserida devemos fornecer uma um prompt de configuração

### Parte do contexto

...

### Chat

Esta parte é apenas a passagem do historico para a ia e o recebimento das perguntas e saida das respostas

## Run

para instalar todas as dependencias nessesarias são

espera o comando abaixo desta linha específica ser concluído

```bash
pip install langchain langchain_chroma langchain_core langchain_text_splitters langchain_openai langchain_community
```

> [!important]
> É recomendado que tenha a chave da api (OPENAI_API_KEY) COLOCADA NAS VARIAVEIS DE AMBIENTE
>
> porem caso não seja possivel altere a linha de [configuração](https://github.com/WebTech-PUC-Minas/assistente-carreiras-gpt/blob/main/lang_chain_cursos.py#L17 "Click para ir para a linha do codigo") da OPENAI
