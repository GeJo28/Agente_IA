# pip install langchain langchain-ollama langchain-community langchain-chroma chromadb pypdf
from langchain_chroma.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
import pyttsx3


CAMINHO_DB = 'database'
engine = pyttsx3.init()

prompt_template = '''
Responda a pergunta do usuário:
{pergunta}

com base nessas informações abaixo:

{base_conhecimento}
'''
def perguntar():
    pergunta = input("Escreva sua pergunta: ")

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
    prompt = prompt.invoke({'pergunta': pergunta, 'base_conhecimento': base_conhecimento})

    # Gerando a resposta atravéz da IA
    modelo = ChatOllama(model='llama3')
    texto_resposta = modelo.invoke(prompt).content
    print(f"Resposta da IA: {texto_resposta}")
    engine.say(texto_resposta)
    engine.runAndWait()

perguntar()