from common import tokenizer,id2tagCIT, id2tagNER
import json
import torch
from transformers import BertTokenizerFast, BertForTokenClassification
import sys


def tag_sentence(sentence):
    model = BertForTokenClassification.from_pretrained('model/bert_multiclass_original',local_files_only=True)
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding='max_length', max_length=128)
    tokens = map(tokenizer.decode, inputs['input_ids'][0])
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

    return [{"token":w, "tag":t} for w,t in zip(words,tags)]


def classify_sentence(sentence):
    model = BertForTokenClassification.from_pretrained('model-just-citation/checkpoint-176',local_files_only=True)
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding='max_length', max_length=128)
    tokens = map(tokenizer.decode, inputs['input_ids'][0])
    logits = model(**inputs).logits
    predictions = torch.argmax(logits, dim=2)
    predicted_token_class = [id2tagCIT[t.item()] for t in predictions[0]]

    words = []
    tags = []

    for x,y,z in zip(tokens,predicted_token_class,inputs.word_ids(0)):
        if z == None: continue
        if z == len(words)-1:
            words[-1] = words[-1] + x[2:]
        else:
            words.append(x)
            tags.append(y)

    return [{"token":w, "tag":t} for w,t in zip(words,tags)]

if __name__=="__main__":
    if sys.argv[1] == "classify":
        print(f"Classifying: {sys.argv[2:]}")
        for x in classify_sentence(sys.argv[2:]):
            print(f"{x['token']} -> {x['tag']}")
    else:
        print(f"Tagging: {sys.argv[1:]}")
        for x in tag_sentence(sys.argv[1:]):
            print(f"{x['token']} -> {x['tag']}")



