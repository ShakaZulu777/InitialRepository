import streamlit as st
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
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

# Define a rugby-focused prompt template
rugby_template = """
You are an expert on rugby rules and rugby game design. Your primary purpose is to:
1. Answer questions about rugby rules with clarity and precision
2. Help users understand rugby situations and procedures
3. Assist in creating and designing rugby games, drills, and training exercises

Use your knowledge to provide accurate, helpful responses focused exclusively on rugby.

Question: {question}
Context: {context}

Answer:
"""

# Initialize Streamlit state
if 'qa_chain' not in st.session_state:
    # Initialize embeddings - using text-embedding-3-small to match 1536 dimensions of your index
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"  # Using compatible embedding model (1536 dimensions)
    )

    # Initialize vector store
    index_name = "pinecone-chatbot20241201"
    vectorstore = PineconeVectorStore.from_existing_index(
        embedding=embeddings,
        index_name=index_name
    )

    # Initialize LLM with the latest model
    llm = ChatOpenAI(
        model="gpt-4o",  # Using the latest GPT-4o model
        temperature=0.2  # Slight temperature to allow for creative game design
    )

    # Create the custom prompt
    rugby_prompt = PromptTemplate(
        template=rugby_template,
        input_variables=["context", "question"]
    )

    # Create QA chain with custom prompt
    st.session_state.qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
        chain_type_kwargs={"prompt": rugby_prompt}
    )

# Set up the Streamlit interface
st.title("Rugby Rules & Game Design Assistant 🏉")
st.subheader("Ask about rugby rules or get help designing rugby games")

# Create a chat interface
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask about rugby rules or game design..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get bot response
    with st.chat_message("assistant"):
        with st.spinner("Analyzing rugby knowledge..."):
            try:
                response = st.session_state.qa_chain.invoke(prompt)
                st.markdown(response['result'])
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": response['result']})
            except Exception as e:
                error_message = f"An error occurred: {str(e)}"
                st.error(error_message)
                # Add error message to chat history
                st.session_state.messages.append({"role": "assistant", "content": error_message})

# Add sidebar with information
with st.sidebar:
    st.markdown("""
    ### Rugby Rules & Game Design Assistant
    
    This specialized rugby assistant can help you with:
    - Understanding official rugby rules and regulations
    - Clarifying specific match situations and referee decisions
    - Learning about game procedures and best practices
    - Designing rugby games, drills, and training exercises
    - Creating rugby strategies and gameplay concepts
    
    ### Example Questions:
    - What are the current rules for a scrum in international rugby?
    - How does a lineout work in the latest World Rugby regulations?
    - Can you design a training game to improve rucking skills?
    - What are some modified rugby games for beginners?
    - Create a drill to practice defensive line speed
    """)
    
    st.divider()
    st.markdown("### Rugby Formats")
    st.markdown("""
    - 15s (Union)
    - 7s (Sevens)
    - 10s (Tens)
    - Touch Rugby
    - Tag Rugby
    """)