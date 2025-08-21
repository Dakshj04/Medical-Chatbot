from flask import Flask,render_template,jsonify,request
from openai import embeddings
from src.helper import download_embeddings
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain 
from langchain_core.prompts import ChatPromptTemplate
from src.prompt import prompt
from dotenv import load_dotenv 
import os 
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate


app=Flask(__name__)
#load the api keys
load_dotenv()
PINECONE_API_KEY=os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")


os.environ["PINECONE_API_KEY"]=PINECONE_API_KEY
os.environ["OPENAI_API_KEY"]=OPENAI_API_KEY
os.environ["OPENAI_BASE_URL"]="https://openrouter.ai/api/v1"
#insert into pincone
index_name="medical-chatbot"
embedding = download_embeddings()
docSearch=PineconeVectorStore.from_existing_index(
    embedding=embedding,
    index_name=index_name
)

retriever = docSearch.as_retriever(search_type="similarity",
        search_kwargs={'k': 3})

chatModel = ChatOpenAI(
    model="gpt-oss-20b:free",
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENAI_API_KEY,
)   
   
#create chains
question_answer_chain= create_stuff_documents_chain(chatModel,prompt)
rag_chain=create_retrieval_chain(retriever,question_answer_chain)


@app.route("/")
def index():
    return render_template('chat.html') 

@app.route("/chat", methods=["GET","POST"])
def chat():
    try:
        data = request.get_json()
        if not data or 'message' not in data:
            return jsonify({"error": "No message provided"}), 400
        
        response = rag_chain.invoke({"input": data['message']})
        return jsonify({
            "response": response["answer"],
            "source": ""
        })
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({
            "error": str(e)
        }), 500

if __name__=="__main__":
    app.run(host="0.0.0.0",port=8080,debug=True)