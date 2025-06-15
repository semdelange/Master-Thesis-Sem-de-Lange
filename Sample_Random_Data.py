from common import tokenizer,id2tagCIT, id2tagNER
import torch
from transformers import BertTokenizerFast, BertForTokenClassification
import sys

import json
import os
import glob
import pandas as pd
from tqdm import tqdm


def tag_sentence(sentence, return_logits=False):
    model = BertForTokenClassification.from_pretrained('model/bert_multiclass_original',local_files_only=True)
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding='max_length', max_length=128)
    tokens = list(map(tokenizer.decode, inputs['input_ids'][0]))
    logits = model(**inputs).logits
    predictions = torch.argmax(logits, dim=2)
    predicted_token_class = [id2tagNER[t.item()] for t in predictions[0]]

    words = []
    tags = []

    for x,y,z in zip(tokens,predicted_token_class,inputs.word_ids(0)):
        if z == None: continue
        if z == len(words)-1:
            words[-1] = words[-1] + x[2:]
        else:
            words.append(x)
            tags.append(y)

    if return_logits:
        return [{"token": w, "tag": t} for w, t in zip(words, tags)], logits
    return [{"token": w, "tag": t} for w, t in zip(words, tags)]


def compute_uncertainty(logits):
    """
    Computes uncertainty based on entropy.
    Higher entropy => more uncertainty.

    Args:
        logits (Tensor): Logits of shape (batch_size, seq_len, num_labels)

    Returns:
        Tensor: Sequence-level uncertainty scores (batch_size,)
    """
    probs = torch.nn.functional.softmax(logits, dim=-1)  # (batch_size, seq_len, num_labels)
    log_probs = torch.log(probs + 1e-12)  # Avoid log(0)
    entropy = -torch.sum(probs * log_probs, dim=-1)  # (batch_size, seq_len)
    sequence_uncertainty = entropy.mean(dim=1)  # Average token entropy per sequence

    return sequence_uncertainty  # (batch_size,)


# Function to read the braces data
def load_braces_data(file_paths):
    braces_data = []

    for file_path in file_paths:
        filename = os.path.basename(file_path)
        state = '-'.join(filename.split('-')[1:-3])  # Extract full state name
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            for line in f:
                parts = line.strip().split(" ", 1)
                if len(parts) == 2:
                    ecli, content = parts
                    braces_data.append((ecli, content, state))

    return braces_data


print("Constructing full dataset")

# Load all files matching pattern "braces-*"
file_paths = glob.glob("All Data Raw/braces-*")
braces_data = load_braces_data(file_paths)

# Place everything in one big dataframe with appropriate columns
df = pd.DataFrame(braces_data, columns=['ECLI', 'Citation', 'State'])
print("Dataset loaded")

# Take a random sample from the dataset --> random_state=42: (Optional) Sets a seed for reproducibility.
citations = df['Citation'].sample(n=5000, random_state=42).tolist()

records = []

for citation in tqdm(citations, desc="Tagging Citations"):
    try:
        result, logits = tag_sentence(citation, return_logits=True)
        tokens = [item["token"] for item in result]
        tags = [item["tag"] for item in result]

        # Compute uncertainty (logits shape: [1, seq_len, num_labels])
        uncertainty = compute_uncertainty(logits)[0].item()  # Extract scalar
    except Exception as e:
        print(f"Error for citation '{citation}': {e}")
        tokens, tags, uncertainty = [], [], 0.0

    records.append({
        "sentence": citation,
        "tokenized_sentence": tokens,
        "predicted_labels": tags,
        "uncertainty": uncertainty
    })

# Sort by uncertainty (descending)
print("Sorting for most uncertain classified citations")
records_sorted = sorted(records, key=lambda r: r["uncertainty"], reverse=True)

# Take top 100 most uncertain
top_uncertain = records_sorted[:100]

# Build final output JSON structure
print("Building JSON structure")
output = {
    "sentences": [r["sentence"] for r in top_uncertain],
    "tokenized_sentence": [r["tokenized_sentence"] for r in top_uncertain],
    "predicted_labels": [r["predicted_labels"] for r in top_uncertain]
}

# Save to file
with open("top_100_uncertain_citations.json", "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

print("Saved top 100 uncertain citations.")

