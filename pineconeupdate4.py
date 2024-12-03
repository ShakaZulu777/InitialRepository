from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Directory path
directory_path = r"C:\Users\13459\Dropbox\Businesses\Rugby Tribe\20241102 RAG Database"

def load_pdfs(directory):
    """Load PDFs using PyPDFLoader instead of Unstructured"""
    documents = []
    for filename in os.listdir(directory):
        if filename.endswith('.pdf'):
            try:
                file_path = os.path.join(directory, filename)
                loader = PyPDFLoader(file_path)
                documents.extend(loader.load())
                print(f"Successfully loaded {filename}")
            except Exception as e:
                print(f"Error loading {filename}: {str(e)}")
    return documents

def main():
    try:
        # Load documents
        print("Loading documents...")
        docs = load_pdfs(directory_path)
        print(f"Loaded {len(docs)} documents")

        # Split documents
        print("Splitting documents...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        split_docs = text_splitter.split_documents(docs)
        print(f"Created {len(split_docs)} chunks")

        # Initialize embeddings
        print("Initializing embeddings...")
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small"
        )

        # Create vector store
        print("Creating vector store...")
        index_name = "pinecone-chatbot20241201"
        vectorstore = PineconeVectorStore.from_documents(
            split_docs, 
            embeddings, 
            index_name=index_name
        )

        # Initialize QA chain
        print("Setting up QA chain...")
        llm = ChatOpenAI(
            model="gpt-4",
            temperature=0
        )
        
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever()
        )

        # Test query
        query = "What are two examples of Front Row Hooker Binds?"
        response = qa.invoke(query)
        print("\nTest Query Result:")
        print(response)

    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()