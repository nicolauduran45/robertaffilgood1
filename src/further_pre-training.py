import glob
import random
from datasets import Dataset, DatasetDict
import pandas as pd
import argparse
from datasets import ClassLabel
import random
import pandas as pd
from IPython.display import display, HTML
from transformers import AutoModelForMaskedLM
from transformers import DataCollatorForLanguageModeling
from transformers import AutoTokenizer
from transformers import EarlyStoppingCallback
from transformers import Trainer, TrainingArguments

parser = argparse.ArgumentParser(description='Further pre-training RoBERTa for affiliation language')
parser.add_argument('-m','--model_path', help='model path', required=True)
parser.add_argument('-t','--train_size', help='url', required=True)
parser.add_argument('-f','--folder_path', help='folder path', required=True)
parser.add_argument('-e','--experiment_name', help='experiment name', required=True)
args = vars(parser.parse_args())

train_size = args['train_size']
model_checkpoint = args['model_path']
folder_path = args['folder_path']
experiment_name = args['experiment_name']
# model_checkpoint = "xlm-roberta-base"

block_size = 128
def show_random_elements(dataset, num_examples=10):
    assert num_examples <= len(dataset), "Can't pick more elements than there are in the dataset."
    picks = []
    for _ in range(num_examples):
        pick = random.randint(0, len(dataset)-1)
        while pick in picks:
            pick = random.randint(0, len(dataset)-1)
        picks.append(pick)

    df = pd.DataFrame(dataset[picks])
    for column, typ in dataset.features.items():
        if isinstance(typ, ClassLabel):
            df[column] = df[column].transform(lambda i: typ.names[i])
    display(HTML(df.to_html()))

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation = True, padding='max_length', max_length=128)

def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
    total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result

#folder_path = '/content/gdrive/MyDrive/SIRIS Projects/04. GRANTS R&D/02-Implementation/2023Building reliable author- and institution-level curation tools for performing accurate analyses on scientific catalogues/2023AffilGood/NER for Affiliation Entity/further pre-training RoBERTa for affiliation language/data'
random_20M_affiliations_raw = pd.read_csv(f"{folder_path}/20M_random_raw-affiliation-string.csv.gz", compression='gzip', header=0, sep=',', quotechar='"', error_bad_lines=False)

# drop duplicates and sample 50k hold out test
random_20M_affiliations_unique = random_20M_affiliations_raw[random_20M_affiliations_raw.RAW_AFFILIATION_STRING.notnull()].rename(columns = {'RAW_AFFILIATION_STRING':'text','WORK_ID':'id'}).drop_duplicates('text').reset_index(drop=True)
affilraw_df = random_20M_affiliations_unique.sample(frac=1, random_state = 42).reset_index(drop=True)
test_df = affilraw_df.sample(50000, random_state = 42)
train_df = affilraw_df.drop(test_df.index).reset_index(drop=True).sample(int(train_size), random_state = 42).reset_index(drop=True)

train_ds = Dataset.from_pandas(train_df.set_index('text'))
test_ds = Dataset.from_pandas(test_df.set_index('text'))

affilraw_dd = DatasetDict({"train":train_ds,"test":test_ds})

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)
tokenized_datasets = affilraw_dd.map(tokenize_function, batched=True, num_proc=4, remove_columns=['text'])

model = AutoModelForMaskedLM.from_pretrained(model_checkpoint)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer,  mlm=True, mlm_probability=0.15)

model_name = model_checkpoint.split("/")[-1]
training_args = TrainingArguments(
    f"/content/gdrive/MyDrive/SIRIS Projects/04. GRANTS R&D/02-Implementation/2023Building reliable author- and institution-level curation tools for performing accurate analyses on scientific catalogues/2023AffilGood/NER for Affiliation Entity/further pre-training RoBERTa for affiliation language/models/affilraw-{model_name}-{experiment_name}",
    evaluation_strategy = "steps",
    save_strategy = "steps",
    per_device_train_batch_size=64,
    per_device_eval_batch_size=16,
    learning_rate=0.0005,#2e-5,
    #weight_decay=0.01,
    #max_steps = 12500,
    eval_steps= 500,
    num_train_epochs=1,
    #gradient_accumulation_steps = 4,
    # warmup_ratio = 0.06,
    #load_best_model_at_end = True,
    #resume_from_checkpoint = True
    #push_to_hub=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator = data_collator,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    #callbacks = [EarlyStoppingCallback(early_stopping_patience = 2)]
)

trainer.train()