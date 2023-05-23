import os
import traceback
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

loader = PyPDFLoader("PDFchat/test_documents/OH HMO EOC (3).pdf")
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
    app.logger.info("Learning...")
    directory="PDFChat/test_documents"
    
    for filename in os.listdir(directory):
        pdf_path = os.path.join(directory, filename)
        loader = PyPDFLoader(pdf_path)
        pages = loader.load_and_split()
        vectordb = Chroma.from_documents(pages, embedding=embeddings, 
                                        persist_directory=".")
    vectordb.persist()
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    pdf_qa = ConversationalRetrievalChain.from_llm(OpenAI(temperature=0.3) , vectordb.as_retriever(), memory=memory)
    app.logger.info("Testing...")
    f = open("PDFchat/test_questions/questions.txt", "r")
    questions=[]
    for line in f:
        if line[:2]=='Q:':
            questions.append(line[3:])
    for query in questions:
        try:
            result=pdf_qa({"question": query})
        except Exception as e:
            app.logger.error(f'An error occured when attempting to answer query: {query}')
            app.logger.error(traceback.format_exc())
        answer=result.get('answer')
        print(f"Q: {query} \n Ans: {answer} \n")
    flash('You can direct your queries about the PDF to /pdfchat')

    return render_template('layout.html')

if __name__ == "__main__":
    app.run(debug=True, host='localhost',port=8000)