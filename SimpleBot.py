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

# Docuent Loading
urls = [
    "https://catalogs.gsu.edu/preview_program.php?catoid=4&poid=1159",
    "https://catalogs.gsu.edu/preview_program.php?catoid=43&poid=12713",
    "https://catalogs.gsu.edu/preview_entity.php?catoid=43&ent_oid=2982",
    "https://www.gsu.edu/program/computer-science-bs/?utm_source=pltitle&utm_medium=cas&utm_content=bs&utm_campaign=program_explorer",
    "https://www.gsu.edu/program/computer-science-ms/?utm_source=pltitle&utm_medium=cas&utm_content=ms&utm_campaign=program_explorer",
    "https://catalogs.gsu.edu/content.php?catoid=42&navoid=5496",
    "https://catalogs.gsu.edu/content.php?catoid=42&navoid=5496#3010-general-information",
    "https://communication.gsu.edu/document/ma-handbook/?wpdmdl=4945&refresh=5faed98232b1d1605294466",
    "https://csds.gsu.edu/?wpdmdl=4939&ind=1620936669195"

]

documents = [WebBaseLoader(url).load() for url in urls]
documentList = [doc for subset in documents for doc in subset]

#Document Split/Chunking
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=450,
    chunk_overlap=100
)

documentSlit = text_splitter.split_documents(documentList)

#vectorize the documents
embeddings = OllamaEmbeddings(model="nomic-embed-text")

vectorstore = SKLearnVectorStore.from_documents(documentSlit, embeddings)

retriever = vectorstore.as_retriever(search_kwargs={"k":4})

