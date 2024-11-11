from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import (
    LoraConfig,
    PeftModel,
    prepare_model_for_kbit_training,
    get_peft_model,
)
import os, torch, wandb, json
from datasets import load_dataset
from trl import SFTTrainer, setup_chat_format

from huggingface_hub import login
hf_token = os.getenv("HF_TOKEN")
login(token = hf_token)

wb_token = os.getenv("WB_TOKEN")

wandb.login(key=wb_token)
run = wandb.init(
    project='Fine-tune Llama-3-8b on SQuAD', 
    job_type="training", 
    anonymous="allow"
)

base_model = "model_path"
new_model = "adapter_save_path"
dataset_name = "SQuAD"


if torch.cuda.get_device_capability()[0] >= 8:
    torch_dtype = torch.bfloat16
else:
    torch_dtype = torch.float16
    
# QLoRA config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch_dtype,
    bnb_4bit_use_double_quant=True,
)

# Load model
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    quantization_config=bnb_config,
    device_map='auto',
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model)

import bitsandbytes as bnb

def find_all_linear_names(model):
    cls = bnb.nn.Linear4bit
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    if 'lm_head' in lora_module_names:  # needed for 16 bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)

modules = find_all_linear_names(model)

# LoRA config
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=modules
)
model, tokenizer = setup_chat_format(model, tokenizer)
model = get_peft_model(model, peft_config)

# Importing the dataset
dataset = load_dataset(dataset_name, split="all")
instruction = "You are a helpful assistant. Your task is to extract the answer directly the provided context."

with open("./data/train-v1.1-with_explanation.json", "r") as f:
    data_exp = json.load(f)
    data_exp_dict = {d['id']: d for d in data_exp}
    
def is_id_in_data_exp_dict(row):
    return str(row['id']) in data_exp_dict

dataset = dataset.filter(is_id_in_data_exp_dict)

def format_chat_template(row):
    context = row['context']
    question = row['question']
    explanation = data_exp_dict[row['id']]["explanation"]
    user_prompt = f"{context}\n {question}\n please ONLY give the direct answer itself without saving any other information including introductory remarks prompts and any periods. Remember, the answer must exist within the context provided. Extract it without adding any extraneous information.\n"
    
    row_json = [
        {"role": "assistant", "content": instruction},
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": explanation}
    ]

    row["text"] = tokenizer.apply_chat_template(row_json, tokenize=False)
    return row

dataset = dataset.map(
    format_chat_template,
    num_proc=4,
)
dataset = dataset.shuffle(seed=65).select(range(3000))

print(dataset['text'][1])
dataset = dataset.train_test_split(test_size=0.1)


# Setting Hyperparamter
#Hyperparamter
training_arguments = TrainingArguments(
    output_dir=new_model,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=2,
    optim="paged_adamw_32bit",
    num_train_epochs=2,
    eval_strategy="steps",
    eval_steps=0.1,
    logging_steps=1,
    warmup_steps=10,
    logging_strategy="steps",
    learning_rate=5e-5,
    fp16=False,
    bf16=False,
    group_by_length=True,
    report_to="wandb"
)

# Setting sft parameters
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    peft_config=peft_config,
    max_seq_length= 256,
    dataset_text_field="text",
    tokenizer=tokenizer,
    args=training_arguments,
    packing= False,
)


model.config.use_cache = False
trainer.train()

wandb.finish()
model.config.use_cache = True

trainer.model.save_pretrained(new_model)