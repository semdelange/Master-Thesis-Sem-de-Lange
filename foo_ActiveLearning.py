from common import tokenizer, tag2idNER, id2tagNER, tag2idCIT, id2tagCIT
import numpy as np
import json
from dataclasses import field
from datasets import Dataset, concatenate_datasets
import pickle

import torch
from torch.utils.data import DataLoader

from transformers import BertTokenizerFast, BertForTokenClassification, AdamW, Trainer, TrainingArguments
from transformers import DataCollatorForTokenClassification


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support

from seqeval.metrics import precision_score, recall_score, f1_score, classification_report
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
import pandas as pd
import matplotlib.pyplot as plt

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


tokenizer = BertTokenizerFast.from_pretrained('bert-base-german-cased')

# Active Learning Parameters
UNCERTAINTY_BATCH_SIZE = 25
AL_ROUNDS = 19
INITIAL_SPLIT = 0.2

def iob_tagging(text, annotations):
    # sentences = sent_tokenize(text)
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


## Old uncertainty computation
# def compute_uncertainty(logits):
#     probs = torch.nn.functional.softmax(logits, dim=-1)
#     entropy = -torch.sum(probs * torch.log(probs + 1e-12), dim=-1)  # shape: (batch_size, seq_len)
#     return entropy.mean(dim=1)  # average entropy per sequence


# def compute_uncertainty(logits):
#     probs = torch.nn.functional.softmax(logits, dim=-1)  # shape: (batch_size, seq_len, num_labels)
#     top2 = torch.topk(probs, k=2, dim=-1)  # Get top-2 class probabilities per token
#     margins = top2.values[..., 0] - top2.values[..., 1]  # p1 - p2, shape: (batch_size, seq_len)
#     token_uncertainty = 1 - margins  # Higher margin = more confident → 1 - margin = more uncertain
#     return token_uncertainty.mean(dim=1)  # Average token uncertainty per sequence

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


if __name__ == "__main__":
    print("This is main")
    torch.set_num_threads(16)
    # print(torch.get_num_threads())
    # exit()
    # nltk.download('punkt')
    # nltk.download('punkt_tab')

    taglist = ['O', 'B-Autor', 'I-Autor', 'B-Aktenzeichen', 'I-Aktenzeichen', 'B-Auflage', 'I-Auflage', 'B-Datum',
               'I-Datum', 'B-Editor', 'B-Gesetz', 'I-Gesetz', 'B-Gericht', 'I-Gericht', 'B-Jahr', 'B-Nummer',
               'I-Nummer', 'B-Randnummer', 'I-Randnummer', 'B-Paragraph', 'I-Paragraph', 'B-Seite-Beginn',
               'I-Seite-Beginn', 'B-Seite-Fundstelle', 'B-Titel', 'I-Titel', 'B-Zeitschrift', 'I-Zeitschrift',
               'I-Editor', 'I-Seite-Fundstelle', 'B-Wort:Auflage', 'I-Wort:Auflage', 'B-Wort:aaO', 'I-Wort:aaO']

    # tag2id = {"O": 0, "B-Autor": 1, "I-Autor": 2, "B-Aktenzeichen": 3, "I-Aktenzeichen": 4, "B-Auflage": 5, "I-Auflage": 6, "B-Datum": 7, "I-Datum": 8, "B-Editor": 9, "B-Gesetz": 10, "I-Gesetz": 11, "B-Gericht": 12, "I-Gericht": 13, "B-Jahr": 14, "B-Nummer": 15, "I-Nummer": 16, "B-Randnummer": 17, "I-Randnummer": 18, "B-Paragraph": 19, "I-Paragraph": 20, "B-Seite-Beginn": 21, "I-Seite-Beginn": 22, "B-Seite-Fundstelle": 23, "B-Titel": 24, "I-Titel": 25, "B-Zeitschrift": 26, "I-Zeitschrift": 27, "I-Editor": 28, "I-Seite-Fundstelle" : 29}
    # id2tag = {0: "O", 1: "B-Autor", 2: "I-Autor", 3: "B-Aktenzeichen", 4: "I-Aktenzeichen", 5: "B-Auflage", 6: "I-Auflage", 7: "B-Datum", 8: "I-Datum", 9: "B-Editor", 10: "B-Gesetz", 11: "I-Gesetz", 12: "B-Gericht", 13: "I-Gericht", 14: "B-Jahr", 15: "B-Nummer", 16: "I-Nummer", 17: "B-Randnummer", 18: "I-Randnummer", 19: "B-Paragraph", 20: "I-Paragraph", 21: "B-Seite-Beginn", 22: "I-Seite-Beginn", 23: "B-Seite-Fundstelle", 24: "B-Titel", 25: "I-Titel", 26: "B-Zeitschrift", 27: "I-Zeitschrift", 28: "I-Editor", 29: "I-Seite-Fundstelle"}
    tag2id = {tag: i for i, tag in enumerate(taglist)}
    id2tag = {i: tag for i, tag in enumerate(taglist)}

    ############################# load training data
    with open('data/1000-aufl_annotations-1.json', 'r') as file:
        data_aufl = json.load(file)

    with open('data/1000-sentences_annotations-3.json', 'r') as file:
        data_sentences = json.load(file)

    input_token_lists = []
    input_tag_lists = []

    datas = [data_aufl, data_sentences]
    # datas = [data_aufl]

    for i in datas:
        for document in i['examples']:
            if document['annotations'] != []:
                text = document['content']
                annotations = document['annotations']
                # if annotations != []:
                token_lists, tag_lists = iob_tagging(text, annotations)
                flattened_token_lists = [item for row in token_lists for item in row]
                flattened_tag_lists = [tagz for columnz in tag_lists for tagz in columnz]
                input_token_lists.append(flattened_token_lists)
                input_tag_lists.append(flattened_tag_lists)

    print(f"Loaded {len(input_token_lists)} samples.")
    json_load_data('data/validated_data.json', input_token_lists, input_tag_lists)
    json_load_data('data/all_regions_validation2.json', input_token_lists, input_tag_lists)

    with open('data/old_documents_labeled.json', 'r') as file:
        data_old = json.load(file)

    tokenized_sentences_old = []
    true_labels_old = []

    for i in data_old['tokenized_sentence']:
        tokenized_sentences_old.append(i)
    for i in data_old['predicted_labels']:
        true_labels_old.append(i)


    with open('data/top_100_uncertain_validated.json', 'r') as file:
        data_extra = json.load(file)

    tokenized_sentences_extra = []
    true_labels_extra = []

    for i in data_extra['tokenized_sentence']:
        tokenized_sentences_extra.append(i)
    for i in data_extra['predicted_labels']:
        true_labels_extra.append(i)


    ################## Prepare data for learning
    dataset = Dataset.from_dict({'tokens': input_token_lists, 'tags': input_tag_lists})
    dataset = dataset.shuffle(seed=42)
    split_idx = int(0.2 * len(dataset))

    labeled_set = dataset.select(range(split_idx))
    unlabeled_set = dataset.select(range(split_idx, len(dataset)))

    val_dataset_dict = {'tokens': tokenized_sentences_old, 'tags': true_labels_old}
    val_dataset = Dataset.from_dict(val_dataset_dict)
    tokenized_val_dataset = val_dataset.map(tokenize_and_align_labels, batched=True)

    extra_dataset_dict = {'tokens': tokenized_sentences_extra, 'tags': true_labels_extra}
    extra_dataset = Dataset.from_dict(extra_dataset_dict)
    tokenized_extra_dataset = extra_dataset.map(tokenize_and_align_labels, batched=True)

    training_args = TrainingArguments(
        use_cpu=True,
        output_dir='./output',
        eval_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=1,
        weight_decay=0.01,
        save_total_limit=100,
        load_best_model_at_end=True,
    )

    # Store full classification reports
    reports = []

    for round_idx in range(AL_ROUNDS):
        if round_idx == 0:
            model = BertForTokenClassification.from_pretrained(
                'bert-base-german-cased', num_labels=len(taglist))
            model_name = "bert-base-german-cased"
        else :
            model = BertForTokenClassification.from_pretrained(
                'model/bert_multiclass', num_labels=len(taglist), local_files_only=True)
            model_name = f"bert_multiclassV{round_idx + 1}"

        print(f"\n=== Active Learning Round {round_idx + 1}/{AL_ROUNDS} ==="
              f"\nUsing model: {model_name}")

        train_test_split = labeled_set.train_test_split(test_size=0.1, shuffle=True)
        train_dataset = train_test_split['train']
        test_dataset = train_test_split['test']
        # train_dataset = labeled_set

        print(f"\nLength of training data = {len(train_dataset)}")
        print(f"Length of unlabeled data = {len(unlabeled_set)}")

        tokenized_train_dataset = train_dataset.map(tokenize_and_align_labels, batched=True)
        tokenized_test_dataset = test_dataset.map(tokenize_and_align_labels, batched=True)

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train_dataset,
            eval_dataset=tokenized_test_dataset
        )
        print("(re)training model")
        trainer.train()
        print("Saving (re)trained model")
        trainer.save_model("model/bert_multiclass")

        # print(f"\nEVAL Results Round {round_idx + 1}/{AL_ROUNDS} ==================")
        # test_results = trainer.evaluate(eval_dataset=tokenized_val_dataset)
        # print(test_results)
        # print("============ =======================================================")

        # Get predictions on the test set
        predictions, labels, _ = trainer.predict(tokenized_val_dataset)

        label_list = [id2tagNER[i] for i in range(len(id2tagNER))]

        true_labels = [[label_list[l] for l, p in zip(label, pred) if l != -100] for label, pred in
                       zip(labels, predictions.argmax(-1))]
        pred_labels = [[label_list[p] for l, p in zip(label, pred) if l != -100] for label, pred in
                       zip(labels, predictions.argmax(-1))]

        ## Flatten true_labels and pred_labels, turning them into a single list of labels instead of a list of label
        ## sequences. That breaks seqeval, which requires sequences to stay grouped.
        # true_labels = [item for sublist in true_labels for item in sublist]
        # pred_labels = [item for sublist in pred_labels for item in sublist]

        report = classification_report(true_labels, pred_labels, output_dict=True)
        reports.append(report)

        print(f"\nRound {round_idx + 1} Averages - Precision: {report['weighted avg']['precision']:.4f}, "
              f"Recall: {report['weighted avg']['recall']:.4f}, "
              f"F1: {report['weighted avg']['f1-score']:.4f}")

        if round_idx == 14:
            unlabeled_set = extra_dataset
            print("\n!!!!! EXTRA DATA HAS BECOME THE REMAINING UNLABELED SET !!!!!\n")

        if len(unlabeled_set) == 0:
            break

        tokenized_unlabeled = unlabeled_set.map(tokenize_and_align_labels, batched=True).remove_columns(
            ['tokens', 'tags'])
        data_collator = DataCollatorForTokenClassification(tokenizer)
        dataloader = torch.utils.data.DataLoader(tokenized_unlabeled, batch_size=32, collate_fn=data_collator)

        model.eval()
        all_logits = []
        with torch.no_grad():
            for batch in dataloader:
                outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
                all_logits.append(outputs.logits)

        logits_tensor = torch.cat(all_logits, dim=0)
        uncertainties = compute_uncertainty(logits_tensor)
        topk_indices = uncertainties.topk(min(UNCERTAINTY_BATCH_SIZE, len(unlabeled_set))).indices.tolist()

        newly_labeled = unlabeled_set.select(topk_indices)
        labeled_set = concatenate_datasets([labeled_set, newly_labeled])
        remaining_indices = [i for i in range(len(unlabeled_set)) if i not in topk_indices]
        unlabeled_set = unlabeled_set.select(remaining_indices)

    print(reports)

    with open('AL_V4_Replace_Unlabeled_With_Extra.pkl', 'wb') as f:
        pickle.dump(reports, f)

    # === Final Evaluation ===
    print("\n=== Final Evaluation on Validation Set ===")
    final_preds, final_labels, _ = trainer.predict(tokenized_val_dataset)
    final_preds = np.argmax(final_preds, axis=2)

    label_list = [id2tag[i] for i in range(len(id2tag))]
    true_flat = [[label_list[l] for l, p in zip(label, pred) if l != -100] for label, pred in
                 zip(final_labels, final_preds)]
    pred_flat = [[label_list[p] for l, p in zip(label, pred) if l != -100] for label, pred in
                 zip(final_labels, final_preds)]

    final_report = classification_report(true_flat, pred_flat)
    print(final_report)

    # # with open("classification_report_active_learning.txt", "w") as f:
    # #     f.write(final_report)
    #
    # trainer.save_model("model_active_learning_final")