import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import BedrockEmbeddings
from langchain.vectorstores import FAISS
from langchain.indexes.vectorstore import VectorstoreIndexCreator
from langchain_aws import BedrockLLM


def hr_index():

    # Load data from pdf url
    data_loader = PyPDFLoader("https://www.upl-ltd.com/images/people/downloads/Leave-Policy-India.pdf")
    # data_test = data_load.load_and_split()
    # print("No. of pages :", len(data_test))
   
    # Spilt data into chunks
    data_split = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", " ", ""], chunk_size = 100, chunk_overlap=10)
    # 
    
    # Create embeddings
    data_embeddings = BedrockEmbeddings(
        credentials_profile_name="default",
        model_id="amazon.titan-embed-text-v2:0"
    )

    # Create vecrod dbb and store embeddings and indexes
    data_index = VectorstoreIndexCreator(
        text_splitter=data_split,
        embedding=data_embeddings,
        vectorstore_cls=FAISS
    )

    db_index = data_index.from_loaders([data_loader])

    return db_index

# Get LLM
def hr_llm():
    llm = BedrockLLM(
        credentials_profile_name="default",
        model_id='amazon.titan-text-lite-v1',
        model_kwargs={
        "max_tokens_to_sample":3000,
        "temperature": 0.1,
        "top_p": 0.9}
    )
    return llm


# Get user prompt, searches for betst match in vector db and send both to llm
def hr_rag_response(index, question):
    hr_rag_query = index.query(question=question, llm=hr_llm())
    return hr_rag_query






