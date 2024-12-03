#pip install --upgrade pinecone-client pinecone-text pinecone-notebooks
#pip install langchain_community langchain_pinecone langchain_openai unstructured langchain-text-splitters

#OPENAI_API_KEY = "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"
#api_key="1bf1a481-2592-49f1-bb60-cdc5a4c001aa"
#PINECONE_API_KEY = "pcsk_5HTNtq_ARDZoWRjCCfYifzKAaH1uHjHypqyw8y5T5RfpX2qXyQfiX9cZW6dq9JoNicCtUE"

# Following tutorial
# https://www.youtube.com/watch?v=Bxj4btI3TzY



from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
import glob

# %%
#!pip uninstall onnx
#!pip install onnx==1.14.0

# %%
loader = DirectoryLoader(r"C:\Users\13459\Dropbox\Businesses\Rugby Tribe\20241102 RAG Database",glob="**/*.pdf")
docs = loader.load()


# %%
docs[0]

# %%
import os

# Set OpenAI API key
#os.environ["OPENAI_API_KEY"] = ""
# Set Pinecone API key
#os.environ["PINECONE_API_KEY"] = ""

# %%
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

index_name = "pinecone-chatbot20241201"

# Split our documents into chunks
text_splitter = RecursiveCharacterTextSplitter()
split_docs = text_splitter.split_documents(docs)


# %%
split_docs[0]

# %%
vectorstore = PineconeVectorStore.from_documents(split_docs, embeddings, index_name=index_name)

# %%
query = "What are two examples of Front Row Hooker Binds?"

similar_docs = vectorstore.similarity_search(query)


# %%
similar_docs[0]


# %%
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA

llm = ChatOpenAI(
    model = "gpt-4o",
    temperature = 0
    
)

qa = RetrievalQA.from_chain_type(
    llm = llm,
    chain_type = "stuff",
    retriever=vectorstore.as_retriever()
)

qa.invoke(query)



