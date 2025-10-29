from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_chroma import Chroma
from loadCV import cv_split

embeddings = OllamaEmbeddings(model='mxbai-embed-large')

vectorstore = Chroma.from_documents(
    documents=cv_split,
    embedding=embeddings,
    persist_directory='./CVBot/chroma_cv'
)