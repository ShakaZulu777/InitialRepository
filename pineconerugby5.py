import streamlit as st
from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores.pinecone import Pinecone as LangChainPinecone
import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

# Load environment variables from .env file
load_dotenv()

# Set your API keys securely
openai_api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")

# Validate API Keys
if not openai_api_key or not pinecone_api_key:
    st.error("API keys are missing. Please check your .env file.")
    st.stop()

try:
    # Initialize Pinecone client
    pc = Pinecone(api_key=pinecone_api_key)

    # Set the index name
    index_name = "pinecone-chatbot2"

    # Check if the index exists; if not, create it
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=1536,
            metric='cosine',
            spec=ServerlessSpec(
                cloud='aws',
                region='us-west-2'
            )
        )

    # Initialize Streamlit state
    if 'qa_chain' not in st.session_state:
        # Initialize embeddings
        embeddings = OpenAIEmbeddings(
            model="text-embedding-ada-002",
            openai_api_key=openai_api_key
        )

        # Initialize vector store using LangChain's Pinecone wrapper
        vectorstore = LangChainPinecone(
            index=pc.Index(index_name),
            embedding=embeddings,
            text_key="text"
        )

        # Initialize retriever
        retriever = vectorstore.as_retriever(
            search_kwargs={"k": 3}  # Retrieve top 3 most relevant chunks
        )

        # Initialize LLM
        llm = ChatOpenAI(
            model_name="gpt-4",
            temperature=0,
            openai_api_key=openai_api_key
        )

        # Create prompt template
        template = """Answer the following question about rugby rules based on the given context. 
        If the context doesn't contain enough information to answer the question, just say that you don't have enough information.

        Context: {context}
        Question: {question}

        Answer the question using only the information provided in the context. If you're not sure, say so."""
        
        prompt = ChatPromptTemplate.from_template(template)

        # Create the chain
        st.session_state.qa_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

except Exception as e:
    st.error(f"An error occurred during initialization: {str(e)}")
    raise e

# Set up the Streamlit interface
st.title("Rugby Rules Assistant 🏉")

# Create a chat interface
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask about rugby rules..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get bot response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = st.session_state.qa_chain.invoke(prompt)
                st.markdown(response)
            except Exception as e:
                response = f"An error occurred while processing your request: {e}"
                st.markdown(response)
            
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

# Add sidebar with information
with st.sidebar:
    st.markdown("""
    ### About This Rugby Rules Assistant
    
    This chatbot can help you with:
    - Understanding rugby rules
    - Clarifying specific situations
    - Learning about game procedures
    
    ### Example Questions:
    - What are the rules for a scrum?
    - How does a lineout work?
    - What constitutes a knock-on?
    - What are the rules for tackling?
    """)