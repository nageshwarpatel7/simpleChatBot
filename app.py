import streamlit as st 
import openai
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate


import os
from dotenv import load_dotenv
load_dotenv()

## Langsmith Tracking
os.environ['LANGCHAIN_API_KEY'] = os.getenv("LANGCHAIN_API_KEY")
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_PROJECT'] = "Simple Q&A Chatbot with GROQ"


## prompt template 
prompt = ChatPromptTemplate.from_messages(
    [
        ("system","You are a helpful assistant. Please response to the user queries"),
        ("user","Question:{question}")
    ]
)

def generate_response(question,api_key,llm,temperature,max_tokens):
    groq_api_key = api_key
    llm = ChatGroq(groq_api_key=groq_api_key,model_name=llm)
    #output_parser = StrOutputParser()
    chain = prompt |llm
    answer = chain.invoke({'question':question})
    
    return answer.content

## Title of the app
st.title("Enhanced Q&A Chatbot with OPENAI")

## Sidebar for settings
st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter your Groq api key:", type="password")

## Dropdown to select various open ai models
llm = st.sidebar.selectbox("Select an Groq API Model",["openai/gpt-oss-20b","gemma2-9b-it","llama-3.1-8b-instant","llama-3.3-70b-versatile"])

## Adjust response parameter
temperature = st.sidebar.slider("Temperature",min_value=0.0,max_value=1.0, value=0.5)
max_tokens = st.sidebar.slider("Max Tokens", min_value=50, max_value=300,value=150)

## Main interface for use input
st.write("Go ahead and ask any question")
user_input = st.text_input("You: ")

if user_input:
    response = generate_response(user_input,api_key,llm,temperature,max_tokens)
    st.write(response)
    
else:
    st.write("Please provide the query")