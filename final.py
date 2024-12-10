import os
import sys
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from openai import OpenAI
load_dotenv()


def main():
    question = sys.argv[1]
    pdf_name = sys.argv[2]
    api_key = os.getenv("OPENAI_API_KEY")

    text = extract_data(pdf_name)
    docs = split_text(text)

    docstorage = vectorize_and_store(docs, api_key)
    response = answer_question(question, api_key, docstorage)

    print(response['result'])
    # return response

def extract_data(pdf_name: str) -> str:
    loader = PyPDFLoader(pdf_name)
    data = loader.load()
    policy_text = ""
    for doc in data:
        if isinstance(doc, dict) and 'text' in doc:
            policy_text += doc['text']
        elif isinstance(doc, str):
            policy_text += doc
        else:
            policy_text += repr(doc)
    return policy_text

def split_text(text):
    ct_splitter = CharacterTextSplitter(separator='.', chunk_size=1000, chunk_overlap=200)
    docs = ct_splitter.split_text(text)
    return docs

def vectorize_and_store(docs, api_key):
    embedding_function = OpenAIEmbeddings(openai_api_key=api_key)
    docstorage = FAISS.from_texts(docs, embedding_function)
    return docstorage

def answer_question(question, api_key, docstorage):
    llm=OpenAI(model_name="gpt-3.5-turbo-instruct", openai_api_key=api_key)
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=docstorage.as_retriever())          
    response = qa.invoke(question)
    return response

main()