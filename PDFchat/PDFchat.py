import os
from langchain.document_loaders import PyPDFLoader 
from langchain.embeddings import OpenAIEmbeddings 
from langchain.vectorstores import Chroma 
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.llms import OpenAI
from flask import Flask, request, Response
from flask import flash,render_template, request, send_from_directory
import json

app = Flask(__name__)

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


@app.route("/pdfchat", methods=['POST'])
def query_chat()->Response:
    if request.content_type == 'application/json':
        try:
            data = json.loads(request.data)
            query=data['query']
        except ValueError:
            print('Failed to decode json')
            return Response(response='Sorry. Failed to decode JSON data', status=415, mimetype='text/plain')
    result = pdf_qa({"question": query})
    #query = "How many fundamental rights do citizens have and what are they?"
    resp={}
    resp['Question']=query
    resp['Answer']=result.get('answer')
    resp=json.dumps({"Response": resp})

    print("Answer:")
    print(result['answer'])
    return Response(response=resp,status=200, mimetype='application/json')



@app.route("/")
def landing_page():
    flash('You can direct your queries about the PDF to /pdfchat')

    return render_template('layout.html')