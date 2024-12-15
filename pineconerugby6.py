import streamlit as st
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
import os
from dotenv import load_dotenv
import speech_recognition as sr
from elevenlabs import generate, play, set_api_key
from pydub import AudioSegment
from pydub.playback import play as pydub_play
from pinecone import Pinecone

load_dotenv()

# Set your API keys
openai_api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
elevenlabs_api_key = os.getenv("ELEVENLABS_API_KEY")
pinecone_environment = os.getenv("PINECONE_ENVIRONMENT", "gcp-starter")

# Validate API Keys
if not openai_api_key:
    st.error("OpenAI API key is missing. Please check your .env file.")
    st.stop()
if not pinecone_api_key:
    st.error("Pinecone API key is missing. Please check your .env file.")
    st.stop()
if not elevenlabs_api_key:
    st.error("ElevenLabs API key is missing. Please check your .env file.")
    st.stop()

# Set the API keys
os.environ["OPENAI_API_KEY"] = openai_api_key
os.environ["PINECONE_API_KEY"] = pinecone_api_key
set_api_key(elevenlabs_api_key)

# Initialize Pinecone with retry logic
@st.cache_resource
def init_pinecone():
    try:
        pc = Pinecone(api_key=pinecone_api_key)
        return pc
    except Exception as e:
        st.error(f"Failed to initialize Pinecone: {str(e)}")
        return None

# Initialize vector store with error handling
@st.cache_resource
def init_vectorstore():
    try:
        pc = init_pinecone()
        if not pc:
            return None
            
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        index_name = "pinecone-chatbot20241201"
        
        # Check if index exists
        if index_name not in pc.list_indexes().names():
            st.error(f"Index '{index_name}' not found in Pinecone")
            return None
            
        vectorstore = PineconeVectorStore.from_existing_index(
            embedding=embeddings,
            index_name=index_name
        )
        return vectorstore
    except Exception as e:
        st.error(f"Error initializing vector store: {str(e)}")
        return None

# Initialize QA chain with modified caching
@st.cache_resource
def create_qa_chain(_vectorstore):
    if not _vectorstore:
        return None
        
    try:
        llm = ChatOpenAI(model="gpt-4", temperature=0, request_timeout=30)
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=_vectorstore.as_retriever(search_kwargs={"k": 3})
        )
        return qa_chain
    except Exception as e:
        st.error(f"Error initializing QA chain: {str(e)}")
        return None

# Initialize core components
if 'qa_chain' not in st.session_state:
    vectorstore = init_vectorstore()
    if vectorstore:
        st.session_state.qa_chain = create_qa_chain(vectorstore)
    else:
        st.error("Failed to initialize the application")
        st.stop()

# Voice capture with timeout and error handling
def capture_voice():
    recognizer = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            st.info("Listening... Please speak your question.")
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
            text = recognizer.recognize_google(audio)
            return text
    except sr.UnknownValueError:
        st.error("Could not understand the audio. Please try again.")
    except sr.RequestError as e:
        st.error(f"Could not request results: {str(e)}")
    except Exception as e:
        st.error(f"Error capturing voice: {str(e)}")
    return None

# Text-to-speech with ElevenLabs using Chris voice
def speak_text(text):
    try:
        audio = generate(
            text=text,
            voice="Chris",  # Changed to use Chris voice
            model="eleven_monolingual_v1"
        )
        play(audio)
    except Exception as e:
        st.error(f"Error in text-to-speech: {str(e)}")

# Process query with timeout handling
def process_query(question):
    try:
        with st.spinner("Thinking..."):
            response = st.session_state.qa_chain.invoke(
                question,
                config={"timeout": 30}
            )
            return response['result']
    except Exception as e:
        st.error(f"Error processing query: {str(e)}")
        return None

# Streamlit UI
st.title("Rugby Rules Assistant 🏉 - With Voice")

if 'messages' not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Voice input handling
if st.button("Click to Ask with Voice"):
    question = capture_voice()
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
            
        answer = process_query(question)
        if answer:
            with st.chat_message("assistant"):
                st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
            speak_text(answer)

# Text input handling
if prompt := st.chat_input("Ask about rugby rules..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
        
    answer = process_query(prompt)
    if answer:
        with st.chat_message("assistant"):
            st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})
        speak_text(answer)

# Sidebar information
with st.sidebar:
    st.markdown("""
    ### About This Rugby Rules Assistant - With Voice
    
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