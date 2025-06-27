import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast, BertForTokenClassification, Trainer, TrainingArguments

tokenizer = BertTokenizerFast.from_pretrained('bert-base-german-cased')

taglistNER = ['O', 'B-Autor', 'I-Autor', 'B-Aktenzeichen', 'I-Aktenzeichen', 'B-Auflage', 'I-Auflage', 'B-Datum', 'I-Datum', 'B-Editor', 'B-Gesetz', 'I-Gesetz', 'B-Gericht', 'I-Gericht', 'B-Jahr', 'B-Nummer', 'I-Nummer', 'B-Randnummer', 'I-Randnummer', 'B-Paragraph', 'I-Paragraph', 'B-Seite-Beginn', 'I-Seite-Beginn', 'B-Seite-Fundstelle', 'B-Titel', 'I-Titel', 'B-Zeitschrift', 'I-Zeitschrift', 'I-Editor', 'I-Seite-Fundstelle', 'B-Wort:Auflage', 'I-Wort:Auflage', 'B-Wort:aaO', 'I-Wort:aaO']

taglistCIT = ['O', 'B-citation', 'I-citation']

tag2idNER = {tag: i for i, tag in enumerate(taglistNER)}
id2tagNER = {i: tag for i, tag in enumerate(taglistNER)}
tag2idCIT = {tag: i for i, tag in enumerate(taglistCIT)}
id2tagCIT = {i: tag for i, tag in enumerate(taglistCIT)}

print(tag2idCIT)


#tag2id = {"O": 0, "B-Autor": 1, "I-Autor": 2, "B-Aktenzeichen": 3, "I-Aktenzeichen": 4, "B-Auflage": 5, "I-Auflage": 6, "B-Datum": 7, "I-Datum": 8, "B-Editor": 9, "B-Gesetz": 10, "I-Gesetz": 11, "B-Gericht": 12, "I-Gericht": 13, "B-Jahr": 14, "B-Nummer": 15, "I-Nummer": 16, "B-Randnummer": 17, "I-Randnummer": 18, "B-Paragraph": 19, "I-Paragraph": 20, "B-Seite-Beginn": 21, "I-Seite-Beginn": 22, "B-Seite-Fundstelle": 23, "B-Titel": 24, "I-Titel": 25, "B-Zeitschrift": 26, "I-Zeitschrift": 27, "I-Editor": 28, "I-Seite-Fundstelle" : 29}
#id2tag = {0: "O", 1: "B-Autor", 2: "I-Autor", 3: "B-Aktenzeichen", 4: "I-Aktenzeichen", 5: "B-Auflage", 6: "I-Auflage", 7: "B-Datum", 8: "I-Datum", 9: "B-Editor", 10: "B-Gesetz", 11: "I-Gesetz", 12: "B-Gericht", 13: "I-Gericht", 14: "B-Jahr", 15: "B-Nummer", 16: "I-Nummer", 17: "B-Randnummer", 18: "I-Randnummer", 19: "B-Paragraph", 20: "I-Paragraph", 21: "B-Seite-Beginn", 22: "I-Seite-Beginn", 23: "B-Seite-Fundstelle", 24: "B-Titel", 25: "I-Titel", 26: "B-Zeitschrift", 27: "I-Zeitschrift", 28: "I-Editor", 29: "I-Seite-Fundstelle"}
 
