import streamlit as st
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
import os
from dotenv import load_dotenv

load_dotenv()

# Set your API keys
# Access the environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")

# Validate API Keys
if not openai_api_key or not pinecone_api_key:
    st.error("API keys are missing. Please check your .env file.")
    st.stop()

# Set the API keys
os.environ["OPENAI_API_KEY"] = openai_api_key
os.environ["PINECONE_API_KEY"] = pinecone_api_key

# Initialize Streamlit state
if 'qa_chain' not in st.session_state:
    # Initialize embeddings
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )

    # Initialize vector store
    index_name = "pinecone-chatbot20241201"
    vectorstore = PineconeVectorStore.from_existing_index(
        embedding=embeddings,
        index_name=index_name
    )

    # Initialize LLM
    llm = ChatOpenAI(
        model="gpt-4",
        temperature=0
    )

    # Create QA chain
    st.session_state.qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever()
    )

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
            response = st.session_state.qa_chain.invoke(prompt)
            st.markdown(response['result'])
            
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response['result']})

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