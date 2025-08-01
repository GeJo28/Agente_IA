from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings


PASTA_BASE = 'base'

def criar_db():
    # carregar_documentos
    documentos = carregar_documentos()
    # dividir os documentos em pedaços de textos (chunks)
    chunks = dividir_chunks(documentos)
    # vetorizar os chunks com o processo de embedding
    vetorizar_chunks(chunks)

def carregar_documentos():
    carregador = PyPDFDirectoryLoader(PASTA_BASE, glob='*.pdf')
    documentos = carregador.load()
    return documentos

def dividir_chunks(documentos):
    separador_documentos = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=500,
        length_function=len,
        add_start_index=True
    )
    chunks = separador_documentos.split_documents(documentos)
    return chunks

def vetorizar_chunks(chunks):
    db = Chroma.from_documents(chunks, OllamaEmbeddings(model='llama3'), persist_directory='database')
    print("Banco de Dados criado com sucesso!")

criar_db()