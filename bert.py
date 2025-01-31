import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding
from datasets import Dataset, DatasetDict, load_dataset
import numpy as np
import evaluate
from peft import LoraConfig, get_peft_model, TaskType

# Check for MPS and set device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")


def preprocess_function(examples):
    tokenized = tokenizer(examples["text"], truncation=True)
    tokenized["labels"] = examples["spam"]
    return tokenized

def run_model(model_file, prompt):
    model = AutoModelForSequenceClassification.from_pretrained(model_file)
    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    predicted_class = torch.argmax(outputs.logits, dim=-1).item()

    print(f"Predicted class: {predicted_class}")


# metrics
accuracy = evaluate.load('accuracy')
auc_score = evaluate.load('roc_auc')

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    probabilities = np.exp(predictions) / np.exp(predictions).sum(-1, keepdims=True)
    positive_class_probs = probabilities[:, 1]

    auc = np.round(auc_score.compute(prediction_scores=positive_class_probs, references=labels)['roc_auc'], 3)
    predicted_classes = np.argmax(predictions, axis=1)

    acc = np.round(accuracy.compute(predictions=predicted_classes, references=labels)['accuracy'], 3)

    return {"Accuracy": acc, "AUC": auc}

# Load and split dataset
data_path = 'fine_tuning/emails.csv'
dataset_dict = load_dataset('csv', data_files=data_path)
train_test_split = dataset_dict['train'].train_test_split(test_size=0.2)
test_valid_split = train_test_split['test'].train_test_split(test_size=0.5)

dataset_dict = {
    'train': train_test_split['train'],
    'validation': test_valid_split['train'],
    'test': test_valid_split['test']
}
dataset_dict = DatasetDict(dataset_dict)

model_path = 'distilbert/distilbert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(model_path)

id2label = {0: 'not spam', 1: 'spam'}
label2id = {'not spam': 0, 'spam': 1}
model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=2, id2label=id2label, label2id=label2id)
for name, _ in model.base_model.named_parameters():
    print(name)

lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,  # seq classification
    r=8,  # rank
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=["q_lin", "v_lin"]  # Apply LoRA to attention layers
)
peft_model = get_peft_model(model, lora_config)
peft_model.to(device)  # Move model to MPS

# Tokenize data
tokenized_data = dataset_dict.map(preprocess_function, batched=True)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Set training parameters
lr = 2e-4
batch_size = 8
num_epochs = 10

training_args = TrainingArguments(
    output_dir='distilbert-email-lora',
    learning_rate=lr,
    per_device_eval_batch_size=batch_size,
    per_device_train_batch_size=batch_size,
    num_train_epochs=num_epochs,
    logging_strategy='epoch',
    eval_strategy='epoch',
    save_strategy='epoch',
    load_best_model_at_end=True
)

trainer = Trainer(
    model=peft_model,
    args=training_args,
    train_dataset=tokenized_data['train'],
    eval_dataset=tokenized_data['validation'],
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

trainer.train()
