import os
import json
import nltk
import matplotlib.pyplot as plt
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# BLEU Score storage file
SCORE_FILE = "bleu_scores.json"

# Load BLEU scores from file
def load_scores():
    try:
        with open(SCORE_FILE, "r") as file:
            data = json.load(file)  # Load JSON data
            if isinstance(data, list):  # Ensure it's a list of dictionaries
                return data
            else:
                print("Warning: Loaded data is not a list. Returning an empty list.")
                return []
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading scores: {e}")
        return []

# Save BLEU scores to file
def save_scores(query, bleu_score):
    existing_scores = load_scores()
    new_entry = {
        "query": query,
        "bleu_score": bleu_score
    }
    existing_scores.append(new_entry)
        
    with open(SCORE_FILE, "w") as f:
        json.dump(existing_scores, f, indent=4)


# Function to compute BLEU Score
def compute_bleu(reference_text, generated_text):
    reference_tokens = [nltk.word_tokenize(reference_text.lower())]  # Tokenized reference
    generated_tokens = nltk.word_tokenize(generated_text.lower())  # Tokenized generated text

    smoothie = SmoothingFunction().method1  # Smoothing to avoid zero BLEU scores
    bleu_score = sentence_bleu(reference_tokens, generated_tokens, smoothing_function=smoothie)

    return bleu_score

# Function to plot BLEU scores

def plot_bleu_scores():
    bleu_scores = load_scores()

    if not bleu_scores:
        print("⚠ No BLEU scores recorded yet! Please interact with the bot first.")
        return

    queries = list(range(1, len(bleu_scores) + 1))

    if isinstance(bleu_scores[0], dict) and "bleu_score" in bleu_scores[0]:
        scores = [score["bleu_score"] for score in bleu_scores]
    else:
        scores = bleu_scores  # Already a list of float values

    plt.figure(figsize=(3.5, 2.5))  # Adjusted for research paper

    plt.plot(queries, scores, marker='o', linestyle='-', color='b', label='BLEU Score')

    plt.xlabel("Query Number", fontsize=12, labelpad=10)  # Increase label font size & padding
    plt.ylabel("BLEU Score", fontsize=12, labelpad=10)
    plt.title("Chatbot BLEU Score", fontsize=13)

    plt.ylim(0, max(scores, default=1) + 0.2)
    plt.legend(fontsize=10)
    plt.grid()

    plt.tight_layout()  # Prevents label cropping
    plt.savefig("bleu_score_plot.png", dpi=600, bbox_inches='tight')
    plt.show()
# from bleu_score import plot_bleu_scores
# plot_bleu_scores()
