from common import tokenizer, tag2idNER, id2tagNER, tag2idCIT, id2tagCIT
import numpy as np
import json
from dataclasses import field
from datasets import Dataset

import torch
from torch.utils.data import DataLoader

from transformers import BertTokenizerFast, BertForTokenClassification, AdamW, Trainer, TrainingArguments

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# from seqeval.metrics import precision_score, recall_score, f1_score, classification_report
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
import pandas as pd

tokenizer = BertTokenizerFast.from_pretrained('bert-base-german-cased')


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


##    def tokenize_and_align_labels(examples):
##        tokenized_inputs = tokenizer(
##            examples['tokens'],
##            is_split_into_words=True,
##            truncation=True,
##            padding='max_length',
##            max_length=128
##        )
##        labels = []
##        for i, label in enumerate(examples['tags']):
##            word_ids = tokenized_inputs.word_ids(batch_index=i)
##            previous_word_idx = None
##            label_ids = []
##            for word_idx in word_ids:
##                if word_idx is None:
##                    label_ids.append(-100)
##                elif word_idx != previous_word_idx:
##                    if word_idx < len(label):
##                        label_ids.append(label_map[label[word_idx]])
##                    else:
##                        label_ids.append(-100)
##                else:
##                    label_ids.append(-100)
##                previous_word_idx = word_idx
##            labels.append(label_ids)
##        tokenized_inputs["labels"] = labels
##        return tokenized_inputs


def json_load_data(filename, input_token_lists, input_tag_lists):
    with open(filename, 'r') as file:
        data_update = json.load(file)

    for i in data_update['tokenized_sentence']:
        input_token_lists.append(i)

    for i in data_update['predicted_labels']:
        input_tag_lists.append(i)


if __name__ == "__main__":
    print("This is main")
    torch.set_num_threads(16)
    print(torch.get_num_threads())
    # exit()
    nltk.download('punkt')
    nltk.download('punkt_tab')

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

    print(
        f"1. Part of the data: initially labelled. Number of Sentences now {len(input_token_lists)}={len(input_tag_lists)}")
    json_load_data('data/validated_data.json', input_token_lists, input_tag_lists)
    print(
        f"2. Part of the data: manually validated. Number of Sentences now {len(input_token_lists)}={len(input_tag_lists)}")
    json_load_data('data/all_regions_validation2.json', input_token_lists, input_tag_lists)
    print(
        f"3. Part of the data: manually validated states. Number of Sentences now {len(input_token_lists)}={len(input_tag_lists)}")

    #############################

    ### add the validation data
    ##
    ###with open('data/all_regions_validation_data.json', 'r') as file:
    ###with open('data/all_regions_validation2.json', 'r') as file:
    ###  all_regions_validation = json.load(file)
    ##

    ####### validation dataset
    # adding old legal text data

    with open('data/old_documents_labeled.json', 'r') as file:
        data_old = json.load(file)

    tokenized_sentences_old = []
    true_labels_old = []

    for i in data_old['tokenized_sentence']:
        tokenized_sentences_old.append(i)

    for i in data_old['predicted_labels']:
        true_labels_old.append(i)

    print(f"RGZ data for validation. Number of Sentences {len(tokenized_sentences_old)}={len(true_labels_old)}")
    print(len(tokenized_sentences_old))
    print(len(true_labels_old))

    ################## Prepare data for learning
    dataset_dict = {'tokens': input_token_lists, 'tags': input_tag_lists}
    dataset = Dataset.from_dict(dataset_dict)

    # Reduce training data to 20%
    small_dataset = dataset.select(range(int(0.2 * len(dataset))))

    train_test_split = small_dataset.train_test_split(test_size=0.1, shuffle=True)
    train_dataset = train_test_split['train']
    test_dataset = train_test_split['test']

    val_dataset_dict = {'tokens': tokenized_sentences_old, 'tags': true_labels_old}
    val_dataset = Dataset.from_dict(val_dataset_dict)

    tokenized_train_dataset = train_dataset.map(tokenize_and_align_labels, batched=True)
    tokenized_val_dataset = val_dataset.map(tokenize_and_align_labels, batched=True)
    tokenized_test_dataset = test_dataset.map(tokenize_and_align_labels, batched=True)

    print('test set:')
    print(tokenized_test_dataset)

    # model = BertForTokenClassification.from_pretrained('bert-base-german-cased', num_labels=len(taglist))
    # model = BertForTokenClassification.from_pretrained('model_pre_ser/bert_multiclass',local_files_only=True)
    # model = BertForTokenClassification.from_pretrained('model/bert_multiclass', local_files_only=True)
    model = BertForTokenClassification.from_pretrained('model/bert_multiclass_20percentdata', local_files_only=True)
    # model = BertForTokenClassification.from_pretrained('results/checkpoint-21',local_files_only=True)

    training_args = TrainingArguments(
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

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_test_dataset
    )

    # trainer.train()
    # print("BERT model trained")
    # trainer.save_model("model/bert_multiclass_20percentdata")
    # trainer.model.save_pretrained("model_pre/bert_multiclass",safe_serialization=False)
    # trainer.model.save_pretrained("model_pre_ser/bert_multiclass",safe_serialization=True)
    # print("BERT model saved")

    print("EVAL Results on Test Data ==========================================")
    test_results = trainer.evaluate(eval_dataset=tokenized_test_dataset)
    print(test_results)
    print("============ =======================================================")

    inputs = tokenizer("vgl. Planck-Brodmann a.a.O.; RGRKomm. 6. Aufl. Anm. 6 zu \u00a7 868 BGB", return_tensors="pt",
                       truncation=True, padding='max_length', max_length=128)

    tokens = map(tokenizer.decode, inputs['input_ids'][0])

    logits = model(**inputs).logits
    predictions = torch.argmax(logits, dim=2)
    predicted_token_class = [id2tag[t.item()] for t in predictions[0]]

    print()

    words = []
    tags = []

    for x, y, z in zip(tokens, predicted_token_class, inputs.word_ids(0)):
        if z == None: continue
        if z == len(words) - 1:
            words[-1] = words[-1] + x[2:]
        else:
            words.append(x)
            tags.append(y)
    for x, y in zip(words, tags):
        print(f"{x} -> {y}")

    # Get predictions on the test set
    predictions, labels, _ = trainer.predict(tokenized_val_dataset)

    label_list = [id2tagNER[i] for i in range(len(id2tagNER))]

    true_labels = [[label_list[l] for l, p in zip(label, pred) if l != -100] for label, pred in
                   zip(labels, predictions.argmax(-1))]
    pred_labels = [[label_list[p] for l, p in zip(label, pred) if l != -100] for label, pred in
                   zip(labels, predictions.argmax(-1))]
    true_labels = [item for sublist in true_labels for item in sublist]
    pred_labels = [item for sublist in pred_labels for item in sublist]

    # Generate classification report
    # report = classification_report(true_labels, pred_labels, labels=label_list)
    report = classification_report(true_labels, pred_labels, output_dict=True)

    print(classification_report(true_labels, pred_labels))

    # === Metric Extraction ===
    metrics = ["precision", "recall", "f1-score"]
    target = "weighted avg"  # could be 'macro avg' or a class label like '0'

    # Print the desired metrics
    print("Metrics for the 20% training data report:")
    for metric in metrics:
        value = report[target][metric]
        print(f"{metric.title()}: {value:.4f}")
