# pip install langchain langchain-ollama langchain-community langchain-chroma chromadb pypdf
from langchain_chroma.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
import pyttsx3


CAMINHO_DB = 'database'
engine = pyttsx3.init()


prompt_template = '''
Histórico da conversa:
{lista_mensagens}

Responda a pergunta do usuário:
{pergunta}

com base nessas informações abaixo:

{base_conhecimento}
'''

def function_messages(lista_mensagens):
    texto = ''
    for mensagem in lista_mensagens:
        if mensagem['role'] == 'user':
            texto += f'Usuário: {mensagem['content']}\n'
        else:
            texto += f'Assistente: {mensagem['content']}\n'
    
    return texto

def perguntar(pergunta, lista_mensagens=[]):
    lista_mensagens.append(
            {'role': 'user', 'content': pergunta}
            )

    # Carregar o banco de dados
    funcao_embedding = OllamaEmbeddings(model='llama3')
    db = Chroma(persist_directory=CAMINHO_DB, embedding_function=funcao_embedding)

    # Comparar a pergunta do usuário (vetorizada) com o meu banco de dados
    resultados = db.similarity_search_with_relevance_scores(pergunta, k=4)
    if len(resultados) == 0 and resultados[0][1] < 0.7:
        print("Não consegui encontrar alguma informação relevante na base")
        return
    
    textos_resultado = []
    for resultado in resultados:
        texto = resultado[0].page_content
        textos_resultado.append(texto)
    
    base_conhecimento = "\n\n----\n\n".join(textos_resultado)
    prompt = ChatPromptTemplate.from_template(prompt_template)
    prompt = prompt.invoke({'pergunta': pergunta, 'base_conhecimento': base_conhecimento, 'lista_mensagens': function_messages(lista_mensagens)})

    # Gerando a resposta atravéz da IA
    modelo = ChatOllama(model='llama3')
    texto_resposta = modelo.invoke(prompt).content
    return texto_resposta
    

lista_mensagens = []
while True:
    pergunta = input("Escreva sua pergunta: ")

    if pergunta == 'sair':
        break
    else:
        resposta = perguntar(pergunta, lista_mensagens)
        lista_mensagens.append(
            {'role': 'assistant', 'content': resposta}
            )
        print(f"Resposta da IA: {resposta}")
        engine.say(resposta)
        engine.runAndWait()