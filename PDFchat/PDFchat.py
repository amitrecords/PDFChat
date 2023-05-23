import os
from langchain.document_loaders import PyPDFLoader 
from langchain.embeddings import OpenAIEmbeddings 
from langchain.vectorstores import Chroma 
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.llms import OpenAI
from flask import Flask


os.environ['OPENAI_API_KEY'] = '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'


pdf_path = "./PDFchat/test_documents/indian_constitution.pdf"
loader = PyPDFLoader(pdf_path)
pages = loader.load_and_split()

embeddings = OpenAIEmbeddings()

vectordb = Chroma.from_documents(pages, embedding=embeddings, 
                                 persist_directory=".")
vectordb.persist()
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
pdf_qa = ConversationalRetrievalChain.from_llm(OpenAI(temperature=0.9) , vectordb.as_retriever(), memory=memory)



def query_chat(query: str)->str:
    #query = "How many fundamental rights do citizens have and what are they?"
    result = pdf_qa({"question": query})
    #print(result)
    print("Answer:")
    print(result['answer'])
    return result['answer']

app = Flask(__name__)

@app.route("/")
def hello_world():
    question="How many fundamental rights do citizens have and what are they?"
    query_chat(question)
    return "<p>Hello, World!</p>"