import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, TrainingArguments
from transformers import Trainer
from transformers import AutoModelForSeq2SeqLM
import torch
import wandb
import gc
import os

os.environ['WANDB_DISABLED']="true"


with open("key.txt") as f:
    wandb.login(key = f.read())
csv = pd.read_csv("data.csv", sep = "\t")
csv = csv.dropna()
tr, va, te = np.split(csv.sample(frac = 1), [int(.6*len(csv)), int(.8*len(csv))])

tokenizer = AutoTokenizer.from_pretrained("./rebel/rebel-large", local_files_only=True)
model = AutoModelForSeq2SeqLM.from_pretrained("./rebel/rebel-large", local_files_only=True)

gc.collect()
torch.cuda.empty_cache()

train_texts = tr['context'].to_list()
test_texts = te['context'].to_list()
validation_texts = va['context'].to_list()

gen_kwargs = {
    "max_length": 256,
    "length_penalty": 0,
    "num_beams": 3,
    "num_return_sequences": 3,
}

train_labels = tr['triplets'].to_list()
test_labels = te['triplets'].to_list()
validation_labels = va['triplets'].to_list()

train_encodings = tokenizer(train_texts, max_length=256, truncation=True, padding=True, return_tensors = 'pt')
validation_encodings = tokenizer(validation_texts, max_length=256, truncation=True, padding=True, return_tensors = 'pt')
test_encodings = tokenizer(test_texts, max_length=256, truncation=True, padding=True, return_tensors = 'pt')

train_encodings1 = tokenizer(train_labels, max_length=256, truncation=True, padding=True, return_tensors = 'pt')
validation_encodings1 = tokenizer(validation_labels, max_length=256, truncation=True, padding=True, return_tensors = 'pt')
test_encodings1 = tokenizer(test_labels, max_length=256, truncation=True, padding=True, return_tensors = 'pt')
    

class RebelDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        #item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        #item['labels'] = torch.tensor(self.labels['input_ids'][idx])
        item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}
        item['labels'] = self.labels['input_ids'][idx].clone().detach()
        return item

    def __len__(self):
        return len(self.labels['input_ids'])

train_dataset = RebelDataset(train_encodings, train_encodings1)
val_dataset = RebelDataset(validation_encodings, validation_encodings1)
test_dataset = RebelDataset(test_encodings, test_encodings1)    
    

training_args = TrainingArguments(
    output_dir='./results_texts',          # output directory
    num_train_epochs=10,             # total number of training epochs
    per_device_train_batch_size=16,  # batch size per device during training, from 16 to 8
    per_device_eval_batch_size=64,   # batch size for evaluation, from 64 to 32
    warmup_steps=200,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay 0.01
    logging_dir='./logs',            # directory for storing logs
    evaluation_strategy = 'epoch',
    logging_strategy = 'epoch',
    do_eval = True,
    do_predict = True,
    load_best_model_at_end=True,
    save_strategy = "epoch",
    optim="adamw_torch" 
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
    #compute_metrics = compute_metrics
)

trainer.train()

model.save_pretrained('./results/2411_text_10epoch_nogroup')