#Import necessary libraries

#Libararies for Documets Loading and Splitting
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
#libraries for Embeddings and Vector Store
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
#Import Embeddings
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import SKLearnVectorStore

