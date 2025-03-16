import getpass
import os

from src.utils import download_embedding_model
from src.logger import logging as log
from src.prompt import *

import pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
import chainlit as cl

from response_time import measure_response_time
from bleu_score import compute_bleu, save_scores


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

'''llm=CTransformers(model="llama-2-7b-chat.ggmlv3.q4_0.bin",
                  model_type="llama",
                  config={'max_new_tokens':256,
                          'temperature':0.8})'''

llm = ChatGroq(
    temperature=0.5,
    groq_api_key="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",
    model_name="deepseek-r1-distill-qwen-32b"
)

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
    log.info(f"Fetching reference response for query: {query}")
    response = qa.invoke(query)
    retrieved_docs = response.get("source_documents", [])

    if retrieved_docs:
        reference_text = " ".join(doc.page_content for doc in retrieved_docs)
        log.info("Reference response retrieved successfully")
    else:
        reference_text = "No reference found"
        log.warning("No reference response found")
    
    return reference_text

# Chainlit Integration
@cl.on_chat_start
async def start():
    log.info("Initializing Chatbot session...")
    cl.user_session.set("qa_chain", qa)
    msg = cl.Message(content="Hi, I am HealthMate. How can I help you today?")
    await msg.send()

@cl.on_message
async def main(message: cl.Message):
    query = message.content.strip()
    log.info(f"Received user query: {query}")
    qa_chain = cl.user_session.get("qa_chain")

    if query.lower() == "exit":
        log.info("User exited the chat")
        await cl.Message(content="Take care of yourself, Goodbye!").send()
        return

    model_name = "Cloud LLM"

    # Measure response time while generating output
    log.info("Measuring response time...")
    response = measure_response_time(model_name, query, qa_chain.invoke)
    
    if not response or "result" not in response:
        log.error("Response generation failed")
        await cl.Message(content="Sorry, I couldn't process your request.").send()
        return

    generated_response = response["result"]
    log.info("Response successfully generated")

    # Get reference response
    reference_response = get_reference_response(query)

    # Compute BLEU score
    bleu_score = compute_bleu(reference_response, generated_response)
    log.info(f"Computed BLEU Score: {bleu_score:.4f}")

    # Store BLEU score in JSON file
    save_scores(query, bleu_score)
    log.info("Saved BLEU score")

    # Display response & BLEU score in the terminal
    print("\n--- Chatbot Response ---")
    print(f"Query: {query}")
    print(f"Generated Response: {generated_response}")
    print(f"✅ BLEU Score: {bleu_score:.4f}\n")

    msg = cl.Message(content=generated_response)
    await msg.send()
    log.info("Response sent to user")
