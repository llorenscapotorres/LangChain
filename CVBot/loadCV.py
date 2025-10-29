from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

pdfLoader = PyPDFLoader(file_path='./CVBot/CV_Llorens_Eng.pdf')
docs = pdfLoader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=20
)
cv_split = splitter.split_documents(documents=docs)