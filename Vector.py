import os
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import CSVLoader
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.indexes import VectorstoreIndexCreator
# from iPython.display import display, Markdown

load_dotenv()
loader = CSVLoader(file_path='data.csv')

index = VectorstoreIndexCreator(
    vectorstore_cls=DocArrayInMemorySearch).from_loaders([loader])

query = "please describe these data"

print(response = index.query(query))
