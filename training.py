from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, TrainingArguments
from transformers import Trainer
from transformers import AutoModelForSeq2SeqLM
import torch
import pandas as pd
import gc



tokenizer = AutoTokenizer.from_pretrained("Babelscape/rebel-large", local_files_only=True)
model = AutoModelForSeq2SeqLM.from_pretrained("Babelscape/rebel-large", local_files_only=True).cuda()

tr = pd.read_csv('train_texts_NOGROUP.csv', sep='\t')
te = pd.read_csv('test_texts_NOGROUP.csv', sep='\t')
va = pd.read_csv('validation_texts_NOGROUP.csv', sep='\t')

#NEW
gc.collect()
torch.cuda.empty_cache()
tr = tr.dropna()
te = te.dropna()
va = va.dropna()

#PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:40
#NEW

train_texts = tr['context'].to_list()
test_texts = te['context'].to_list()
validation_texts = va['context'].to_list()

for i in train_texts:
    if type(i) != str:
        print(i)

gen_kwargs = {
    "max_length": 256,
    "length_penalty": 0,
    "num_beams": 3,
    "num_return_sequences": 3,
}

#train_labels = tr['train']['triplets']
#test_labels = te['train']['triplets']
#validation_labels = va['train']['triplets']

train_labels = tr['triplets'].to_list()
test_labels = te['triplets'].to_list()
validation_labels = va['triplets'].to_list()


#max_length added and return_tensors
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
        """check if my changes broke anything"""
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
    
#do_eval added
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


#def compute_metrics(p):
#    pred, labels = p
#    pred = np.argmax(pred[0], -1)
#    #assert len(pred) == len(labels)
#
#
#    #accuracy = accuracy_score(y_true=labels, y_pred=pred)
#    #recall = recall_score(y_true=labels, y_pred=pred)
#    #precision = precision_score(y_true=labels, y_pred=pred)
#    #f1 = f1_score(y_true=labels, y_pred=pred)
#
#    #return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}
#    return accuracy.compute(predictions=pred, references=labels)



trainer = Trainer(
    model=model,                         # the instantiated Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,         # training dataset
    eval_dataset=val_dataset            # evaluation dataset
    #compute_metrics = compute_metrics
)

trainer.train()
#trainer.save_model('./results_texts/0911_text_1epoch')

model.save_pretrained('./results_texts/2411_text_10epoch_nogroup')