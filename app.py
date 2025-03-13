import getpass
from src.utils import download_embedding_model
from src.logger import logging as log
from langchain_pinecone import PineconeVectorStore
import pinecone
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
import chainlit as cl
import json
import os
import time
from response_time import measure_response_time
from src.prompt import *

import nltk
from nltk.translate.bleu_score import sentence_bleu

SCORE_FILE = "bleu_scores.json"

log.info("Getting credentials")
if not os.getenv("PINECONE_API_KEY"):
    os.environ["PINECONE_API_KEY"] = getpass.getpass("Enter your Pinecone API key: ")
pinecone_api_key = os.environ.get("PINECONE_API_KEY")

# Initialise the Pinecone
log.info("Initialising Pinecone")
pc = pinecone.Pinecone(api_key=pinecone_api_key)

# Connect to pinecone index
log.info("Connecting to Pinecone Index...")
index_name = "medical-chatbot"
index = pc.Index(index_name)
log.info(f"Connected to Pinecone Index: {index_name}.")

# Download embeddings model
log.info("Downloading embeddings model")
embeddings = download_embedding_model()
log.info("Embeddings model downloaded")

# Initialising vector store
log.info("Initialising vector store")
vector_store = PineconeVectorStore(index=index, embedding=embeddings)

log.info("Creating prompt")
PROMPT=PromptTemplate(template=prompt_template, input_variables=["context", "question"])
chain_type_kwargs={"prompt": PROMPT}

log.info("Creating LLM model")

llm=CTransformers(model="llama-2-7b-chat.ggmlv3.q4_0.bin",
                  model_type="llama",
                  config={'max_new_tokens':256,
                          'temperature':0.8})

''''llm = ChatGroq(
    temperature=0.5,
    groq_api_key="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",
    model_name="llama-3.3-70b-versatile"
)'''

# Transforming the vector store into a retriever
log.info("Query by turning into retriever")
retriever = vector_store.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 1, "score_threshold": 0.5},
)

log.info("RetrievalQA")
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs=chain_type_kwargs
)

# Function to extract reference response from source documents
def get_reference_response(query):
    response = qa.invoke(query)
    retrieved_docs = response.get("source_documents", [])

    if retrieved_docs:
        reference_text = " ".join(doc.page_content for doc in retrieved_docs)
    else:
        reference_text = "No reference found"

    return reference_text

# Function to compute BLEU score
def compute_bleu_score(reference, candidate):
    if reference == "No reference found":
        return 0  # No meaningful reference available

    reference_tokens = [reference.split()]
    candidate_tokens = candidate.split()
    
    return sentence_bleu(reference_tokens, candidate_tokens)

# Function to store BLEU scores in JSON (without generated response)
def save_bleu_score(query, bleu_score):
    try:
        with open(SCORE_FILE, "r") as f:
            bleu_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        bleu_data = []

    bleu_data.append({
        "query": query,
        "bleu_score": bleu_score
    })

    with open(SCORE_FILE, "w") as f:
        json.dump(bleu_data, f, indent=4)

# Chainlit Integration
@cl.on_chat_start
async def start():
    log.info("Initializing Chatbot...")
    cl.user_session.set("qa_chain", qa)
    msg = cl.Message(content="Hi, I am HealthMate. How can I help you today?")
    await msg.send()

@cl.on_message
async def main(message: cl.Message):
    query = message.content.strip()
    qa_chain = cl.user_session.get("qa_chain")

    if query.lower() == "exit":
        await cl.Message(content="Take care of yourself, Goodbye!").send()
        return

    model_name = "Local LLM"

    # Measure response time while generating output
    response = measure_response_time(model_name, query, qa_chain.invoke)

    if not response or "result" not in response:
        await cl.Message(content="Sorry, I couldn't process your request.").send()
        return

    generated_response = response["result"]

    # Get reference response
    reference_response = get_reference_response(query)

    # Compute BLEU score
    bleu_score = compute_bleu_score(reference_response, generated_response)

    # Store BLEU score in JSON file (without generated response)
    save_bleu_score(query, bleu_score)

    # Display response & BLEU score in the terminal
    print("\n--- Chatbot Response ---")
    print(f"Query: {query}")
    print(f"Generated Response: {generated_response}")
    print(f"✅ BLEU Score: {bleu_score:.4f}\n")

    msg = cl.Message(content=generated_response)
    await msg.send()
