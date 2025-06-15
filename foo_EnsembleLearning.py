from common import tokenizer, tag2idNER, id2tagNER, tag2idCIT, id2tagCIT
import numpy as np
import json
from dataclasses import field
from datasets import Dataset, concatenate_datasets
from collections import Counter

import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.nn.functional as F
from TorchCRF import CRF

import itertools

from transformers import BertTokenizerFast, BertForTokenClassification, AdamW, Trainer, TrainingArguments
from transformers import RobertaTokenizerFast, RobertaForTokenClassification, DataCollatorForTokenClassification

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_recall_fscore_support

import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
import pandas as pd

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

tokenizer = BertTokenizerFast.from_pretrained('bert-base-german-cased')
roberta_tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base", add_prefix_space=True)


class BiLSTMTagger(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, tagset_size, dropout=0.3):
        super(BiLSTMTagger, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embedding_dim, hidden_dim, num_layers=1,
            bidirectional=True, batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        self.hidden2tag = nn.Linear(hidden_dim * 2, tagset_size)

        # Optional: CRF layer for better sequence modeling (can be added later)
        self.crf = CRF(tagset_size)

    def forward(self, input_ids, attention_mask=None):
        embeds = self.embedding(input_ids)  # (batch, seq_len, embed_dim)
        lstm_out, _ = self.lstm(embeds)     # (batch, seq_len, hidden_dim*2)
        lstm_out = self.dropout(lstm_out)
        tag_space = self.hidden2tag(lstm_out)  # (batch, seq_len, tagset_size)
        return tag_space  # logits


def iob_tagging(text, annotations):
    sentences = [text]
    all_tokens = []
    all_tags = []

    for sentence in sentences:
        tokens = word_tokenize(sentence)
        tags = ['O'] * len(tokens)
        sentence_start = text.index(sentence)
        token_positions = []
        position = sentence_start
        for token in tokens:
            position = text.find(token, position)
            token_positions.append((position, position + len(token)))
            position += len(token)

        for annotation in annotations:
            start, end = annotation['start'], annotation['end']
            label = annotation['tag']
            start_token = next((i for i, pos in enumerate(token_positions) if pos[0] <= start < pos[1]), None)
            end_token = next((i for i, pos in enumerate(token_positions) if pos[0] < end <= pos[1]), None)

            if start_token is not None and end_token is not None and start_token < len(tags) and end_token < len(tags):
                tags[start_token] = f'B-{label}'
                for i in range(start_token + 1, end_token + 1):
                    tags[i] = f'I-{label}'

        all_tokens.append(tokens)
        all_tags.append(tags)

    return all_tokens, all_tags


def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples['tokens'], is_split_into_words=True, truncation=True, padding='max_length',
                                 max_length=128)
    labels = []
    for i, label in enumerate(examples['tags']):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(tag2id[label[word_idx]])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs


def json_load_data(filename, input_token_lists, input_tag_lists):
    with open(filename, 'r') as file:
        data_update = json.load(file)

    for i in data_update['tokenized_sentence']:
        input_token_lists.append(i)

    for i in data_update['predicted_labels']:
        input_tag_lists.append(i)


def compute_metrics(p):
    predictions = np.argmax(p.predictions, axis=2)
    true_preds, true_labels = [], []
    for pred, label in zip(predictions, p.label_ids):
        for p_, l_ in zip(pred, label):
            if l_ != -100:
                true_preds.append(p_)
                true_labels.append(l_)

    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, true_preds, average='weighted', zero_division=0)
    acc = accuracy_score(true_labels, true_preds)
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}


def compute_class_weights(tag_lists, tag2id):
    flat_tags = [tag for tag_seq in tag_lists for tag in tag_seq]
    tag_counts = Counter(flat_tags)
    total = sum(tag_counts.values())
    weights = [0.0] * len(tag2id)
    for tag, idx in tag2id.items():
        count = tag_counts.get(tag, 1)  # Avoid zero
        weights[idx] = total / (len(tag2id) * count)
    return torch.tensor(weights, dtype=torch.float)

# def encode_tokens(tokens_list, tags_list, max_len=128):
#     encoded_inputs = []
#     encoded_tags = []
#
#     for tokens, tags in zip(tokens_list, tags_list):
#         input_ids = tokenizer.convert_tokens_to_ids(tokens)
#         input_ids = input_ids[:max_len] + [0] * (max_len - len(input_ids))
#         label_ids = [tag2id[tag] for tag in tags[:max_len]] + [-100] * (max_len - len(tags))
#         encoded_inputs.append(input_ids)
#         encoded_tags.append(label_ids)
#     return torch.tensor(encoded_inputs), torch.tensor(encoded_tags)

def encode_tokens(tokens_list, tags_list, pad_token_id=0, max_len=128):
    input_ids_list = []
    tag_ids_list = []

    for tokens, tags in zip(tokens_list, tags_list):
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        tag_ids = [tag2id.get(tag, tag2id['O']) for tag in tags]

        # Padding
        input_ids = input_ids[:max_len]
        tag_ids = tag_ids[:max_len]
        pad_len = max_len - len(input_ids)
        input_ids += [pad_token_id] * pad_len
        tag_ids += [-100] * pad_len

        input_ids_list.append(input_ids)
        tag_ids_list.append(tag_ids)

    return torch.tensor(input_ids_list), torch.tensor(tag_ids_list)


def train_bilstm(model, input_ids, labels, epochs=20, batch_size=16):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    class_weights = compute_class_weights(input_tag_lists, tag2id).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-100)

    dataloader = DataLoader(TensorDataset(input_ids, labels), batch_size=batch_size, shuffle=True)
    for epoch in range(epochs):
        total_loss = 0.0
        all_preds = []
        all_labels = []

        for input_batch, label_batch in dataloader:
            input_batch, label_batch = input_batch.to(device), label_batch.to(device)
            optimizer.zero_grad()
            logits = model(input_batch)
            logits = logits.view(-1, logits.shape[-1])
            label_batch = label_batch.view(-1)
            loss = loss_fn(logits, label_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

            preds = torch.argmax(logits, dim=1)
            mask = label_batch != -100
            all_preds.extend(preds[mask].cpu().numpy())
            all_labels.extend(label_batch[mask].cpu().numpy())

        f1 = f1_score(all_labels, all_preds, average="macro")
        print(f"[BiLSTM] Epoch {epoch + 1}/{epochs} - Loss: {total_loss:.4f} - F1: {f1:.4f}")


if __name__ == "__main__":
    print("This is main")
    torch.set_num_threads(16)

    taglist = ['O', 'B-Autor', 'I-Autor', 'B-Aktenzeichen', 'I-Aktenzeichen', 'B-Auflage', 'I-Auflage', 'B-Datum',
               'I-Datum', 'B-Editor', 'B-Gesetz', 'I-Gesetz', 'B-Gericht', 'I-Gericht', 'B-Jahr', 'B-Nummer',
               'I-Nummer', 'B-Randnummer', 'I-Randnummer', 'B-Paragraph', 'I-Paragraph', 'B-Seite-Beginn',
               'I-Seite-Beginn', 'B-Seite-Fundstelle', 'B-Titel', 'I-Titel', 'B-Zeitschrift', 'I-Zeitschrift',
               'I-Editor', 'I-Seite-Fundstelle', 'B-Wort:Auflage', 'I-Wort:Auflage', 'B-Wort:aaO', 'I-Wort:aaO']

    tag2id = {tag: i for i, tag in enumerate(taglist)}
    id2tag = {i: tag for i, tag in enumerate(taglist)}

    with open('data/1000-aufl_annotations-1.json', 'r') as file:
        data_aufl = json.load(file)

    with open('data/1000-sentences_annotations-3.json', 'r') as file:
        data_sentences = json.load(file)

    input_token_lists = []
    input_tag_lists = []

    datas = [data_aufl, data_sentences]
    for i in datas:
        for document in i['examples']:
            if document['annotations'] != []:
                text = document['content']
                annotations = document['annotations']
                token_lists, tag_lists = iob_tagging(text, annotations)
                flattened_token_lists = [item for row in token_lists for item in row]
                flattened_tag_lists = [tagz for columnz in tag_lists for tagz in columnz]
                input_token_lists.append(flattened_token_lists)
                input_tag_lists.append(flattened_tag_lists)

    print(f"1. Part of the data: initially labelled. Number of Sentences now {len(input_token_lists)}={len(input_tag_lists)}")
    json_load_data('data/validated_data.json', input_token_lists, input_tag_lists)
    print(f"2. Part of the data: manually validated. Number of Sentences now {len(input_token_lists)}={len(input_tag_lists)}")
    json_load_data('data/all_regions_validation2.json', input_token_lists, input_tag_lists)
    print(f"3. Part of the data: manually validated states. Number of Sentences now {len(input_token_lists)}={len(input_tag_lists)}")

    with open('data/old_documents_labeled.json', 'r') as file:
        data_old = json.load(file)
    tokenized_sentences_old = []
    true_labels_old = []
    for i in data_old['tokenized_sentence']:
        tokenized_sentences_old.append(i)
    for i in data_old['predicted_labels']:
        true_labels_old.append(i)

    print(f"RGZ data for validation. Number of Sentences {len(tokenized_sentences_old)}={len(true_labels_old)}")

    dataset_dict = {'tokens': input_token_lists, 'tags': input_tag_lists}
    dataset = Dataset.from_dict(dataset_dict)

    ################### ADDING EXTRA DATA ###################
    with open('data/top_100_uncertain_validated.json', 'r') as file:
        data_extra = json.load(file)
    tokenized_sentences_extra = []
    true_labels_extra = []
    for i in data_extra['tokenized_sentence']:
        tokenized_sentences_extra.append(i)
    for i in data_extra['predicted_labels']:
        true_labels_extra.append(i)

    extra_dataset_dict = {'tokens': tokenized_sentences_extra, 'tags': true_labels_extra}
    extra_dataset = Dataset.from_dict(extra_dataset_dict)

    dataset = concatenate_datasets([dataset, extra_dataset])
    #########################################################

    train_test_split = dataset.train_test_split(test_size=0.2, shuffle=True)
    train_dataset = train_test_split['train']
    test_dataset = train_test_split['test']

    val_dataset_dict = {'tokens': tokenized_sentences_old, 'tags': true_labels_old}
    val_dataset = Dataset.from_dict(val_dataset_dict)

    tokenized_train_dataset = train_dataset.map(tokenize_and_align_labels, batched=True)
    tokenized_test_dataset = test_dataset.map(tokenize_and_align_labels, batched=True)
    tokenized_val_dataset = val_dataset.map(tokenize_and_align_labels, batched=True)

    data_collator = DataCollatorForTokenClassification(tokenizer)

    bert_model = BertForTokenClassification.from_pretrained('model/bert_multiclass_original', local_files_only=True)
    roberta_model = RobertaForTokenClassification.from_pretrained("model/roberta_multiclass", local_files_only=True, num_labels=len(taglist))
    # roberta_model = RobertaForTokenClassification.from_pretrained("roberta-base", local_files_only=True, num_labels=len(taglist))

    # For fine-tune training the BERT model
    training_args_bert = TrainingArguments(
        use_cpu=True,
        output_dir='./output',
        eval_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=4,
        weight_decay=0.01,
        save_total_limit=100,
        load_best_model_at_end=True,
    )
    trainer_bert = Trainer(
        model=bert_model,
        args=training_args_bert,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_test_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )
    # print("Training BERT model")
    # trainer_bert.train()
    # print("BERT model trained")

    # For fine-tune training the RoBERTa model
    training_args_roberta = TrainingArguments(
        use_cpu=True,
        output_dir="./output-roberta",
        eval_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=8,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        # warmup_steps=100,
        # logging_dir='./logs',
        # logging_steps=50,
    )
    trainer_roberta = Trainer(
        model=roberta_model,
        args=training_args_roberta,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_test_dataset,
        data_collator=data_collator,
        # processing_class=roberta_tokenizer,     # "processing_class" used to be called "tokenizer" instead.
        compute_metrics=compute_metrics
    )
    # print("Training RoBERTa model")
    # trainer_roberta.train()
    # print("RoBERTa model trained")
    # trainer_roberta.save_model("model/roberta_multiclass")
    # print("RoBERTa model saved")

    # BiLSTM Training & Loading
    bilstm_weights_path = "model/bilstm_weights.pt"
    bilstm_model = BiLSTMTagger(vocab_size=30522, embedding_dim=128, hidden_dim=256, tagset_size=len(taglist))

    if not os.path.exists(bilstm_weights_path):
        print("Training BiLSTM model...")
        train_input_ids, train_label_ids = encode_tokens(input_token_lists, input_tag_lists)
        train_bilstm(bilstm_model, train_input_ids, train_label_ids)
        os.makedirs("model", exist_ok=True)
        torch.save(bilstm_model.state_dict(), bilstm_weights_path)
        print("Saved BiLSTM weights.")
    else:
        bilstm_model.load_state_dict(torch.load(bilstm_weights_path))
        print("Loaded BiLSTM weights.")


    ## Old ensemble_predict function
    # def ensemble_predict(val_dataset_tokenized):
    #     label_list = [id2tagNER[i] for i in range(len(id2tagNER))]
    #
    #     # Get BERT predictions
    #     bert_predictions, bert_labels, _ = trainer_bert.predict(val_dataset_tokenized)
    #     bert_probs = torch.tensor(bert_predictions)  # shape: (batch, seq_len, num_labels)
    #
    #     # Get RoBERTa predictions
    #     roberta_predictions, roberta_labels, _ = trainer_roberta.predict(val_dataset_tokenized)
    #     roberta_probs = torch.tensor(roberta_predictions)  # shape: (batch, seq_len, num_labels)
    #
    #     # Prepare BiLSTM input
    #     input_tokens_batch = val_dataset_tokenized['tokens']
    #     bilstm_input_ids, _ = encode_tokens(input_tokens_batch, val_dataset_tokenized['tags'])
    #     bilstm_model.eval()
    #     with torch.no_grad():
    #         bilstm_logits = bilstm_model(bilstm_input_ids)
    #         bilstm_probs = F.softmax(bilstm_logits, dim=-1)
    #
    #     # Compute weighted average probabilities
    #     avg_probs = (
    #             1.0 * F.softmax(bert_probs, dim=-1) +
    #             0.0 * F.softmax(roberta_probs, dim=-1) +
    #             0.0 * bilstm_probs
    #     )
    #
    #     predictions = torch.argmax(avg_probs, dim=-1).numpy()
    #     labels = np.array(val_dataset_tokenized["labels"])
    #
    #     # Convert predictions and labels to tag strings
    #     true_labels = [[label_list[l] for l, p in zip(label, pred) if l != -100]
    #                    for label, pred in zip(labels, predictions)]
    #     pred_labels = [[label_list[p] for l, p in zip(label, pred) if l != -100]
    #                    for label, pred in zip(labels, predictions)]
    #
    #     # Flatten
    #     true_labels = [item for sublist in true_labels for item in sublist]
    #     pred_labels = [item for sublist in pred_labels for item in sublist]
    #
    #     return true_labels, pred_labels

    def ensemble_predict(val_dataset_tokenized, temperatures=(1.0, 1.0, 1.0), weight_grid=None):
        """
        Ensemble prediction with soft-voting and temperature scaling.
        Uses grid search to fine-tune weights for BERT, RoBERTa, and BiLSTM.
        """
        label_list = [id2tagNER[i] for i in range(len(id2tagNER))]

        # === Step 1: Get predictions from all models ===
        bert_logits, _, _ = trainer_bert.predict(val_dataset_tokenized)
        roberta_logits, _, _ = trainer_roberta.predict(val_dataset_tokenized)

        input_tokens_batch = val_dataset_tokenized['tokens']
        bilstm_input_ids, _ = encode_tokens(input_tokens_batch, val_dataset_tokenized['tags'])
        bilstm_model.eval()
        with torch.no_grad():
            bilstm_logits = bilstm_model(bilstm_input_ids)

        # Apply temperature scaling
        T_bert, T_roberta, T_bilstm = temperatures
        bert_probs = F.softmax(torch.tensor(bert_logits) / T_bert, dim=-1)
        roberta_probs = F.softmax(torch.tensor(roberta_logits) / T_roberta, dim=-1)
        bilstm_probs = F.softmax(bilstm_logits / T_bilstm, dim=-1)

        # === Step 2: Grid Search to find best weights ===
        if weight_grid is None:
            weight_grid = [(w1, w2, w3) for w1, w2, w3 in itertools.product(
                [0.1 * i for i in range(11)],
                [0.1 * i for i in range(11)],
                [0.1 * i for i in range(11)]
            ) if abs((w1 + w2 + w3) - 1.0) < 1e-6]

        best_f1 = -1
        best_weights = (1.0, 0.0, 0.0)
        labels = np.array(val_dataset_tokenized["labels"])

        for w1, w2, w3 in weight_grid:
            avg_probs = w1 * bert_probs + w2 * roberta_probs + w3 * bilstm_probs
            predictions = torch.argmax(avg_probs, dim=-1).numpy()

            true_labels_flat = []
            pred_labels_flat = []

            for label, pred in zip(labels, predictions):
                true = [label_list[l] for l, p in zip(label, pred) if l != -100]
                pred_ = [label_list[p] for l, p in zip(label, pred) if l != -100]
                true_labels_flat.extend(true)
                pred_labels_flat.extend(pred_)

            f1 = f1_score(true_labels_flat, pred_labels_flat, average='macro')
            if f1 > best_f1:
                best_f1 = f1
                best_weights = (w1, w2, w3)

        # === Step 3: Final prediction using best weights ===
        w1, w2, w3 = best_weights
        final_probs = w1 * bert_probs + w2 * roberta_probs + w3 * bilstm_probs
        predictions = torch.argmax(final_probs, dim=-1).numpy()

        true_labels_flat = []
        pred_labels_flat = []

        for label, pred in zip(labels, predictions):
            true = [label_list[l] for l, p in zip(label, pred) if l != -100]
            pred_ = [label_list[p] for l, p in zip(label, pred) if l != -100]
            true_labels_flat.extend(true)
            pred_labels_flat.extend(pred_)

        print(f"Best weights found: BERT={w1}, RoBERTa={w2}, BiLSTM={w3} | Best F1: {best_f1:.4f}")
        return true_labels_flat, pred_labels_flat

    # Results
    print("EVAL Results (Ensemble) =======================================================")
    true_labels, pred_labels = ensemble_predict(tokenized_val_dataset)
    report = classification_report(true_labels, pred_labels, output_dict=True, zero_division=0)
    print(classification_report(true_labels, pred_labels, zero_division=0))

    # === Metric Extraction ===
    metrics = ["precision", "recall", "f1-score"]
    target = "weighted avg"  # could be 'macro avg' or a class label like '0'

    # Print the desired metrics
    print("Metrics for the ensemble learning report:")
    for metric in metrics:
        value = report[target][metric]
        print(f"{metric.title()}: {value:.4f}")
