print("Starting...")
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

loader = PyPDFLoader("Preetam_Sharma_Resume_Inn.pdf")
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = splitter.split_documents(docs)

print("\n===== SYSTEM METRICS =====")
print("Total Pages:", len(docs))
print("Total Chunks:", len(chunks))
print("Chunk Size: 500")
print("Chunk Overlap: 50")
print("Embedding Model: all-MiniLM-L6-v2")
print("Vector Store: FAISS")
print("LLM: Gemini 2.5 Flash")
print("==========================\n")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = FAISS.from_documents(chunks, embeddings)

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)

print("RAG Ready")

while True:
    question = input("\nAsk Question: ")

    if question.lower() == "exit":
        break

    docs = vectorstore.similarity_search(question, k=3)

    context = "\n".join([doc.page_content for doc in docs])

    prompt = f"""
    Answer only from the context below.

    Context:
    {context}

    Question:
    {question}
    """

    response = llm.invoke(prompt)

    print("\nAnswer:")
    print(response.content)