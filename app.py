import os
os.environ["USER_AGENT"] = "GSUChatbot/1.0"

from flask import Flask, request, jsonify, render_template
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import SKLearnVectorStore
from langchain_core.documents import Document

app = Flask(__name__)

# Load Documents
urls = [
    "https://catalogs.gsu.edu/preview_program.php?catoid=4&poid=1159",
    "https://catalogs.gsu.edu/preview_program.php?catoid=43&poid=12713",
    "https://catalogs.gsu.edu/preview_entity.php?catoid=43&ent_oid=2982",
    "https://www.gsu.edu/program/computer-science-bs/?utm_source=pltitle&utm_medium=cas&utm_content=bs&utm_campaign=program_explorer",
    "https://www.gsu.edu/program/computer-science-ms/?utm_source=pltitle&utm_medium=cas&utm_content=ms&utm_campaign=program_explorer",
    "https://catalogs.gsu.edu/content.php?catoid=42&navoid=5496",
    "https://catalogs.gsu.edu/content.php?catoid=42&navoid=5496#3010-general-information",
    "https://csds.gsu.edu/undergraduate-program/",
    "https://csds.gsu.edu/graduate-program/",
]

documents = [WebBaseLoader(url).load() for url in urls]
documentList = [doc for subset in documents for doc in subset]

# Clean Documents
cleanedDoc = [
    Document(page_content=doc.page_content.strip(), metadata=doc.metadata)
    for doc in documentList
    if doc.page_content.strip()
]

# Split into Chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150,
    separators=["\n\n", "\n", ".", " ", ""],
)
documentSplit = text_splitter.split_documents(cleanedDoc)

# Build Vector Store
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = SKLearnVectorStore.from_documents(documentSplit, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

# Build RAG Chain
prompt = PromptTemplate(
    template="""You are a helpful assistant for Georgia State University.
Answer ONLY using the documents provided below from GSU's official website.
If the documents contain relevant information, use it to give a helpful answer
even if the answer is not stated word for word — for example, you can infer
that a 120 credit hour program typically takes 4 years.
If there is truly no relevant information at all, say: "I don't know."
Do NOT answer anything unrelated to GSU academics, or anything illegal, political, or harmful.

Documents:
{documents}

Question:
{question}

Answer (max 3 sentences):
""",
    input_variables=["question", "documents"],
)

langModel = ChatOllama(model="llama3.1", temperature=0)
ragChain = prompt | langModel | StrOutputParser()


# Routes
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json(force=True, silent=True)
    if not data:
        return jsonify({"answer": "Invalid request."})

    question = data.get("question", "").strip()
    if not question:
        return jsonify({"answer": "Please enter a question."})

    docs = retriever.invoke(question)
    if not docs:
        return jsonify({"answer": "I don't know based on the available documents."})

    context = "\n\n".join(doc.page_content for doc in docs)
    answer = ragChain.invoke({"question": question, "documents": context})
    return jsonify({"answer": answer})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
