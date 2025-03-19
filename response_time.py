import time
import json
import os
import matplotlib.pyplot as plt

# Define file to store response times
RESPONSE_TIME_FILE = "response_times.json"

# Load existing response times (if any)
def load_response_times():
    if os.path.exists(RESPONSE_TIME_FILE):
        with open(RESPONSE_TIME_FILE, "r") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return []
    return []

# Save response times in a well-formatted structure
def save_response_times(response_times):
    with open(RESPONSE_TIME_FILE, "w") as f:
        json.dump(response_times, f, indent=4)  # Indented JSON for better readability

# Function to measure and store response time
def measure_response_time(model_name, query, response_function):
    response_times = load_response_times()  # Load previous times
    
    start_time = time.time()  # Start timer
    response = response_function(query)  # Get response from LLM
    end_time = time.time()  # End timer

    elapsed_time = end_time - start_time  # Calculate response time

    # Store result in a well-formatted JSON structure
    response_entry = {
        "model": model_name,
        "query": query,
        "response_time": round(elapsed_time, 4)  # Round for better readability
    }

    response_times.append(response_entry)
    save_response_times(response_times)  # Save to JSON file
    
    return response

# Function to plot response time for Local LLM vs Cloud LLM
def plot_response_times():
    response_times = load_response_times()

    if not response_times:
        print("⚠ No response time data found!")
        return

    local_times = [entry["response_time"] for entry in response_times if entry.get("model") == "Local LLM"]
    cloud_times = [entry["response_time"] for entry in response_times if entry.get("model") == "Cloud LLM"]

    if not local_times and not cloud_times:
        print("⚠ No valid data for Local or Cloud LLM found!")
        return

    queries_local = list(range(1, len(local_times) + 1))
    queries_cloud = list(range(1, len(cloud_times) + 1))

    plt.figure(figsize=(4, 3))  # Adjusted for better visualization

    if local_times:
        plt.plot(queries_local, local_times, marker='o', linestyle='-', color='r', label="Local LLM")
    if cloud_times:
        plt.plot(queries_cloud, cloud_times, marker='s', linestyle='-', color='g', label="Cloud LLM")

    plt.xlabel("Query Number", fontsize=12, labelpad=10)
    plt.ylabel("Response Time (seconds)", fontsize=12, labelpad=10)

    # Increase space for the title
    plt.title("Response Time Comparison: Locally stored LLM vs Cloud LLM", fontsize=13)

    plt.legend(fontsize=10)
    plt.grid()

    plt.tight_layout(rect=[0, 0, 1, 0.85])

    plt.savefig("response_time_comparison_fixed.png", dpi=600, bbox_inches="tight")
    plt.show()
