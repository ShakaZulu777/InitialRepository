from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os
from PIL import Image
import io
import base64
import fitz  # PyMuPDF for better PDF image extraction
from openai import OpenAI

# Load environment variables
load_dotenv()

# Directory path
directory_path = r"C:\Users\13459\Dropbox\Businesses\Rugby Tribe\RAG Database"

# Initialize the OpenAI client for image understanding
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def extract_images_from_pdf(pdf_path):
    """Extract images from PDF using PyMuPDF (fitz)"""
    image_descriptions = []
    try:
        pdf_document = fitz.open(pdf_path)
        for page_num, page in enumerate(pdf_document):
            image_list = page.get_images(full=True)
            
            for img_index, img in enumerate(image_list):
                xref = img[0]
                base_image = pdf_document.extract_image(xref)
                image_bytes = base_image["image"]
                
                # Convert image bytes to base64 for OpenAI API
                base64_image = base64.b64encode(image_bytes).decode('utf-8')
                
                # Get image description using GPT-4 Vision
                try:
                    response = client.chat.completions.create(
                        model="gpt-4o",
                        messages=[
                            {"role": "system", "content": "You are a rugby expert. Describe this rugby-related image in detail, focusing on techniques, positions, rules or gameplay elements shown."},
                            {"role": "user", "content": [
                                {"type": "text", "text": "What does this rugby image show? Focus on technical details."},
                                {"type": "image_url", "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }}
                            ]}
                        ],
                        max_tokens=300
                    )
                    description = response.choices[0].message.content
                    image_descriptions.append(f"Image on page {page_num+1}: {description}")
                    print(f"Processed image {img_index+1} on page {page_num+1} of {pdf_path}")
                except Exception as e:
                    print(f"Error processing image {img_index+1} on page {page_num+1}: {str(e)}")
    
    except Exception as e:
        print(f"Error extracting images from {pdf_path}: {str(e)}")
    
    return image_descriptions

def load_pdfs(directory):
    """Load PDFs using PyPDFLoader and extract image content"""
    documents = []
    for filename in os.listdir(directory):
        if filename.endswith('.pdf'):
            try:
                file_path = os.path.join(directory, filename)
                
                # 1. Extract and process images from the PDF
                print(f"Extracting images from {filename}...")
                image_descriptions = extract_images_from_pdf(file_path)
                
                # 2. Load text content using PyPDFLoader
                loader = PyPDFLoader(file_path)
                pdf_documents = loader.load()
                
                # 3. Add image descriptions to the document metadata
                for doc in pdf_documents:
                    # Add the filename to help with context
                    doc.metadata["source"] = filename
                    
                documents.extend(pdf_documents)
                
                # 4. Create an additional document for the image descriptions if they exist
                if image_descriptions:
                    from langchain_core.documents import Document
                    image_doc = Document(
                        page_content="\n\n".join(image_descriptions),
                        metadata={"source": filename, "content_type": "image_descriptions"}
                    )
                    documents.append(image_doc)
                
                print(f"Successfully loaded {filename} with text and {len(image_descriptions)} images")
                
            except Exception as e:
                print(f"Error loading {filename}: {str(e)}")
    return documents

def main():
    try:
        # Load documents
        print("Loading documents and processing images...")
        docs = load_pdfs(directory_path)
        print(f"Loaded {len(docs)} documents including image descriptions")

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

        # Create rugby-specific prompt
        rugby_prompt = PromptTemplate(
            template="""
            You are an expert on rugby rules, techniques, and training. Your knowledge includes understanding
            of rugby diagrams, field positions, and visual training materials.
            
            Use the following pieces of context (including any image descriptions) to answer the question. 
            If you don't know the answer, just say that you don't know, don't try to make up an answer.
            
            Context: {context}
            
            Question: {question}
            
            Answer:
            """,
            input_variables=["context", "question"]
        )

        # Initialize QA chain with the latest model
        print("Setting up QA chain...")
        llm = ChatOpenAI(
            model="gpt-4o",  # Using the latest GPT-4o model
            temperature=0.1  # Slight temperature for more natural responses
        )
        
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
            chain_type_kwargs={"prompt": rugby_prompt}
        )

        # Test query
        query = "What are two examples of Front Row Hooker Binds? Include any visual examples if available."
        response = qa.invoke(query)
        print("\nTest Query Result:")
        print(response)

    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()