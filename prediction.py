from re import U
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import pandas as pd
import torch
import json
from tqdm import tqdm
import torch
import gc 
import os

"""Script to make predictions for test data

Concept:
for each article in test data
    
    make predictions
    check if duplicates
    append to list

write json file

Next step: Calculate metrics between gold and prediction file
"""

#Class to create dataset
class RebelDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels['input_ids'][idx])  #to-do attention mask
        return item

    def __len__(self):
        return len(self.labels['input_ids'])

#From Huggingface to retrieve triplets from predicted outputs
def extract_triplets(text):
    triplets = []
    relation, subject, relation, object_ = '', '', '', ''
    text = text.strip()
    current = 'x'
    for token in text.replace("<s>", "").replace("<pad>", "").replace("</s>", "").split():
        if token == "<triplet>":
            current = 't'
            if relation != '':
                triplets.append({'head': subject.strip(), 'type': relation.strip(),'tail': object_.strip()})
                relation = ''
            subject = ''
        elif token == "<sub>":
            current = 's'
            if relation != '':
                triplets.append({'head': subject.strip(), 'type': relation.strip(),'tail': object_.strip()})
            object_ = ''
        elif token == "<obj>":
            current = 'o'
            relation = ''
        else:
            if current == 't':
                subject += ' ' + token
            elif current == 's':
                object_ += ' ' + token
            elif current == 'o':
                relation += ' ' + token
    if subject != '' and relation != '' and object_ != '':
        triplets.append({'head': subject.strip(), 'type': relation.strip(),'tail': object_.strip()})
    return triplets

#function used to write the respective json files
def write_list(a_list, filename):
    with open("%s.json"%filename, "w") as fp:
        json.dump(a_list, fp)

#params
gen_kwargs = {
    "max_length": 256,
    "length_penalty": 0,
    "num_beams": 3,
    "num_return_sequences": 3,
}

PATH = './results/2411_text_10epoch_nogroup'
file_exists = os.path.exists(PATH)
print(file_exists)

model = AutoModelForSeq2SeqLM.from_pretrained(PATH, local_files_only=False).cuda()
tokenizer = AutoTokenizer.from_pretrained("Babelscape/rebel-large", local_files_only = True)
print('here')

#import test_dataset
te = pd.read_csv('test_texts_NOGROUP.csv', sep='\t')
te = te.dropna()

test_texts = te['context'].to_list()
test_labels = te['triplets'].to_list()
test_encodings = tokenizer(test_texts, max_length=256, truncation=True, padding=True, return_tensors = 'pt')
test_encodings1 = tokenizer(test_labels, max_length=256, truncation=True, padding=True, return_tensors = 'pt')


#Create test datset
test_dataset = RebelDataset(test_encodings, test_encodings1) 

#Tokenize text
model_inputs = test_dataset.encodings

#To be filled arrays
triplets_cleaned = []

#Prediction Loop
for i in tqdm(range(len(te))):

    # Generate tokens and labels
    generated_tokens = model.generate(
        torch.unsqueeze(model_inputs["input_ids"][i], 0).to(model.device),
        attention_mask=torch.unsqueeze(model_inputs["attention_mask"][i], 0).to(model.device),
        **gen_kwargs,
    )  

    # Extract text
    decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=False)

    # Clean predictions
    triplet_list_sample = []
    triplets_cleaned_sample = []

    for idx, sentence in enumerate(decoded_preds):
        triplet_list_sample.append(extract_triplets(sentence))
    
    for trip in triplet_list_sample:
        if trip != [] and trip not in triplets_cleaned_sample:
            triplets_cleaned_sample.append(trip)
    
    triplets_cleaned.append(triplets_cleaned_sample)


# Export 
write_list(triplets_cleaned, 'predictions_2411_text_text_textfinetune')