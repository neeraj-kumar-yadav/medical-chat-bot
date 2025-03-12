#from rouge import Rouge
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
import matplotlib.pyplot as plt
import json


from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import nltk



from src.prompt import *
import os


SCORE_FILE = "bleu_scores.json"

# Load BLEU scores if the file exists
def load_scores():
    global bleu_scores
    if os.path.exists(SCORE_FILE):
        with open(SCORE_FILE, "r") as f:
            try:
                bleu_scores = json.load(f)
            except json.JSONDecodeError:
                bleu_scores = []
    else:
        bleu_scores = []

# Save BLEU scores after each interaction
def save_scores():
    with open(SCORE_FILE, "w") as f:
        json.dump(bleu_scores, f)

# Load BLEU scores at the start
load_scores()


log.info("Getting credentials")
if not os.getenv("PINECONE_API_KEY"):
    os.environ["PINECONE_API_KEY"] = getpass.getpass("Enter your Pinecone API key: ")
pinecone_api_key = os.environ.get("PINECONE_API_KEY")

# Initialise the Pinecone
log.info("Initialise the Pinecone")
pc = pinecone.Pinecone(api_key=pinecone_api_key)

# Connect to pinecone index
log.info("Connecting to Pinecone Index...")
index_name = "medical-chatbot"
index = pc.Index(index_name)
log.info(f"Connected to Ponecone Index: {index_name}.")

# Download embeddings model from huggingface
log.info("Downloading embeddings model")
embeddings = download_embedding_model()
log.info("Embeddings model downloaded")

# Initialising vector store
log.info("Initialising vector store")
vector_store = PineconeVectorStore(index=index, embedding=embeddings)

log.info("Creating prompt")
PROMPT=PromptTemplate(template=prompt_template, input_variables=["context", "question"])
chain_type_kwargs={"prompt": PROMPT}


log.info("creating llm model")
'''llm=CTransformers(model="llama-2-7b-chat.ggmlv3.q4_0.bin",
                  model_type="llama",
                  config={'max_new_tokens':512,
                          'temperature':0.8})'''
llm = ChatGroq(
        temperature=0.8,
        groq_api_key = "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",
        model_name="llama-3.3-70b-versatile"
    )

#gemma2-9b-it
# Query by turning into retriever
# Transforming the vector store into a retriever for easier usage in chains.
log.info("Query by turning into retriever")
retriever = vector_store.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 1, "score_threshold": 0.5},
)
log.info("RetrivalQA")
qa=RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff", 
    retriever=retriever,
    return_source_documents=True, 
    chain_type_kwargs=chain_type_kwargs
    )

# Function to extract reference response from source documents
def get_reference_response(query):
    """
    Extracts the reference response from the retrieved source documents.
    """
    response = qa.invoke(query)  # Get response including retrieved documents
    retrieved_docs = response.get("source_documents", [])

    if retrieved_docs:
        reference_text = " ".join(doc.page_content for doc in retrieved_docs)
    else:
        reference_text = "No reference found"  

    return reference_text

# Function to compute ROUGE Score
def compute_bleu(reference_text, generated_text):
    reference_tokens = [nltk.word_tokenize(reference_text.lower())]  # List of references
    generated_tokens = nltk.word_tokenize(generated_text.lower())  # Tokenized generated text

    # Apply Smoothing to avoid zero BLEU for short responses
    smoothie = SmoothingFunction().method1  # Handles missing n-grams smoothly

    bleu_score = sentence_bleu(reference_tokens, generated_tokens, smoothing_function=smoothie)
    return bleu_score

# Chainlit Integration
@cl.on_chat_start
async def start():
    log.info("Initializing Chatbot...")

    # Store retriever and QA chain in user session
    cl.user_session.set("qa_chain", qa)

    # Send initial message
    msg = cl.Message(content="Hi, I am HealthMate. How can I help you today?")
    await msg.send()

# Lists to store reference and generated texts for evaluation
reference_texts = []
generated_texts = []
bleu_scores = []
@cl.on_message
async def main(message: cl.Message):
    query = message.content.strip()

    # Retrieve the QA chain from user session
    qa_chain = cl.user_session.get("qa_chain")

    if query.lower() == "exit":
        await cl.Message(content="Take care of yourself, Goodbye!").send()
        return

    # Generate response
    response = qa_chain.invoke(query)
    generated_response = response["result"]

    # Reference text
    reference_text = get_reference_response(query)  
    reference_texts.append(reference_text)

    # Compute BLEU score
    smoothie = SmoothingFunction().method1
    bleu_score = sentence_bleu([reference_text.split()], generated_response.split(), smoothing_function=smoothie)

    # Store and save BLEU score
    bleu_scores.append(bleu_score)
    save_scores()  # Save BLEU scores after every response

    # Debugging: Print BLEU score
    print(f"BLEU Score for this response: {bleu_score:.4f}")

    await cl.Message(content=generated_response).send()
# ========== Generate BLEU Score Plot ==========
def plot_bleu_scores():
    load_scores()  # Load stored BLEU scores before plotting

    if not bleu_scores:
        print("⚠ No BLEU scores recorded yet! Please interact with the bot first.")
        return

    queries = list(range(1, len(bleu_scores) + 1))

    plt.figure(figsize=(10, 5))
    plt.plot(queries, bleu_scores, marker='o', linestyle='-', color='b', label='BLEU Score')

    # Labels and Title
    plt.xlabel("Query Number")
    plt.ylabel("BLEU Score")
    plt.title("Chatbot Performance Over Multiple Queries")
    plt.ylim(0, 1)
    plt.legend()
    plt.grid()

    # Save as image for research paper
    plt.savefig("chatbot_bleu_performance.png", dpi=300, bbox_inches='tight')
    plt.show()

# ========= ⚡ How to Run ⚡ =========
# ✅ First, interact with the chatbot.
# ✅ Then, manually call `plot_bleu_scores()` from VS Code Terminal:
# >>> from app import plot_bleu_scores
# >>> plot_bleu_scores()
#from app import bleu_scores, plot_bleu_scores
#print("All BLEU Scores:", bleu_scores)
#plot_bleu_scores()