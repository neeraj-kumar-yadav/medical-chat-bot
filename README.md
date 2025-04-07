# Medical Chatbot

## Overview

This project is a **Medical Chatbot** that retrieves information from medical PDFs using **vector embeddings** stored in **Pinecone**. It processes user queries, retrieves relevant information, and evaluates its responses using the **BLEU score** metric. The chatbot is built using **LangChain, Chainlit, Pinecone, and Groq Cloud API**.

## Features

- **Medical Knowledge Retrieval**: Uses PDFs as a knowledge base.
- **Vector Embeddings**: Queries are matched with vectorized document chunks.
- **BLEU Score Evaluation**: Assesses chatbot performance over multiple interactions.
- **Interactive Chat Interface**: Built using **Chainlit**.
- **Logging & Debugging**: BLEU scores are stored and plotted for performance analysis.

## Installation

### Prerequisites

Ensure you have **Python 3.8+** installed. Then, install the required dependencies:

```sh
pip install -r requirements.txt
```

### Environment Variables

Create a `.env` file in the project root and add:

```sh
GROQ_API_KEY=your_groq_api_key
PINECONE_API_KEY=your_pinecone_api_key
```

## Running the Chatbot

To start the chatbot, run:

```sh
chainlit run app.py
```

## Evaluating BLEU Score

The chatbot stores BLEU scores for responses. You can visualize performance using:

```sh
python plot_bleu.py
```

This generates a graph (`chatbot_bleu_performance.png`) showing chatbot accuracy over time.

## Repository Structure

```
├── data/
├── logs/
├── src/
│   ├── __init__.py
│   ├── utils.py
│   ├── prompt.py
├── research/
│   ├── trials.ipynb
├── app.py                 # Main script
├── store_index.py         # Stores vector embeddings
├── plot_bleu.py           # BLEU score plotting script
├── requirements.txt       # Project dependencies
├── setup.py               # Project setup script
├── .env                   # API keys (not committed)
```

## Contributing

Feel free to fork and improve the project. Open a pull request for major changes!

## License

This project is licensed under the **MIT License**.

