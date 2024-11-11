import torch
import math
import json
import re
import os
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from multiprocessing import Manager
import torch.nn.functional as F

qa_prompt = (
    '\nPlease only give the answer without saying any other information.\n'
)
manager = Manager()
res = manager.dict()


instruction = "You are a helpful assistant. Your task is to extract the answer directly the provided context."

def generate_answer_finetune_explanation(model_path, data, res, start_index=None, end_index=None, device='cpu'):
    file_path = './result/finetune_explanation/phi-3.5-mini-instruct-explanation-tuned.json'
    model = AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model.cuda('cuda:' + str(device))

    for article in tqdm(data[start_index:end_index], desc=f'Processing from {start_index} to {end_index}'):
        for p in tqdm(article['paragraphs'], desc=f'Processing paragraphs'):
            context = p['context']
            for qa in p['qas']:
                qid = qa['id']
                question = qa['question']
                
                instruction = "You are a helpful assistant. Your task is to extract the answer directly the provided context."
                user_prompt = f"{context}\n {question}\n please ONLY give the direct answer itself without saving any other information including introductory remarks prompts and any periods. Remember, the answer must exist within the context provided. Extract it without adding any extraneous information.\n"
                messages = [
                    {"role": "assistant", "content": instruction},
                    {"role": "user", "content": user_prompt},
                ]
                
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                
                model_inputs = tokenizer([text], return_tensors="pt").to(device)
                
                outputs = model.generate(
                    **model_inputs,
                    max_new_tokens=128,
                    output_scores=True,
                    return_dict_in_generate=True
                )
                
                generated_tokens = outputs.sequences
                
                scores = outputs.scores
                
                response_text = tokenizer.decode(generated_tokens[0], skip_special_tokens=True).split('assistant\n')[-1]

                token_probs = []
                tokens = []
                collect_tokens = False
                
                hash_1, hash_2 = find_hash_positions_with_regex(response_text)
                if hash_1 is None or hash_2 is None:
                    token_probs.append(math.log(0.5))
                    res[qid] = {
                        'res': response_text,
                        'prob': math.exp(sum(token_probs)),
                    }
                    continue
                
                for i, token_id in enumerate(generated_tokens[0][model_inputs['input_ids'].size(-1):]):
                    token_str = tokenizer.decode([token_id], clean_up_tokenization_spaces=False)
                    
                    token_start = response_text.find(token_str)
                    token_end = token_start + len(token_str)
                    
                    if token_start >= hash_1 and token_end <= hash_2:
                        collect_tokens = True
                    elif token_start > hash_2:
                        collect_tokens = False
                        break
                    
                    if collect_tokens and token_str != "#":
                        logits = scores[i]
                        log_probs = F.log_softmax(logits, dim=-1)
                        token_log_prob = log_probs[0, token_id].item()
                        token_probs.append(token_log_prob)
                        tokens.append(token_str)
                        
                    # Update token_start
                    token_start = token_end 

                res[qid] = {
                    'res': response_text,
                    'prob': math.exp(sum(token_probs)),
                }
                
        # Save result
        with open(file_path, 'w') as f:
            json.dump(dict(res), f, indent=4)
            f.close()
                    
                    
def generate_answer_finetune(model_path, data, res, start_index=None, end_index=None, device='cpu', mode='zero-shot'):
    file_path = './result/phi-3.5-mini-instruct-one-shot.json'
    model = AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    model.cuda('cuda:' + str(device))

    for article in tqdm(data[start_index:end_index], desc=f'Processing from {start_index} to {end_index}'):
        for p in tqdm(article['paragraphs'], desc=f'Processing paragraphs'):
            context = p['context']
            for qa in p['qas']:
                qid = qa['id']
                question = qa['question']
                
                if mode == 'zero-shot':
                    user_prompt = f"{context}\n {question}\n please ONLY give the direct answer itself without saving any other information including introductory remarks prompts and any periods. Remember, the answer must exist within the context provided. Extract it without adding any extraneous information.\n"
                    instruction = "You are a helpful assistant. Your task is to extract the answer directly the provided context."
                    messages = [
                        {"role": "system", "content": instruction},
                        {"role": "user", "content": user_prompt},
                    ]
                
                elif mode == 'one-shot':
                    instruction = """
                        You are a helpful assistant. Your task is to extract the answer directly the provided context. Please ONLY give the direct answer itself without saving any other information including introductory remarks prompts and any periods. Remember, the answer must exist within the context provided. Extract it without adding any extraneous information.\n
                    """
                    example = """
                        Context: "Architecturally, the school has a Catholic character. Atop the Main Building's gold dome is a golden statue of the Virgin Mary. Immediately in front of the Main Building and facing it, is a copper statue of Christ with arms upraised with the legend \"Venite Ad Me Omnes\". Next to the Main Building is the Basilica of the Sacred Heart. Immediately behind the basilica is the Grotto, a Marian place of prayer and reflection. It is a replica of the grotto at Lourdes, France where the Virgin Mary reputedly appeared to Saint Bernadette Soubirous in 1858. At the end of the main drive (and in a direct line that connects through 3 statues and the Gold Dome), is a simple, modern stone statue of Mary."
                        Question: "To whom did the Virgin Mary allegedly appear in 1858 in Lourdes France?"
                    """
                    example_answer = """Answer: "Saint Bernadette Soubirous"""
                    user_prompt = f"Context: {context}\n Question:{question}\n"
                    messages = [
                        {"role": "assistant", "content": instruction},
                        {"role": "user", "content": example},
                        {"role": "assistant", "content": example_answer},
                        {"role": "user", "content": user_prompt},
                    ]
                
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                
                model_inputs = tokenizer([text], return_tensors="pt").to(device)
                
                outputs = model.generate(
                    **model_inputs,
                    max_new_tokens=256,
                )
                
                
                response_text = tokenizer.decode(outputs[0], skip_special_tokens=True).split('\n')[-1]

                if mode == 'zero-shot':
                    res[qid] = response_text.split('\n')[0]   # Use for zero-shot
                elif mode == 'one-shot':    
                    res[qid] = response_text
                
                torch.cuda.empty_cache()
                
        with open(file_path, 'w') as f:
            json.dump(dict(res), f, indent=4)
            f.close()

def find_hash_positions_with_regex(text):
    matches = [m.start() for m in re.finditer(r'#', text)]
    if len(matches) < 2:
        return matches + [None] * (2 - len(matches))

    return matches[0], matches[1]


if __name__ == '__main__':
    # Run on multiple GPUs
    num_workers = 4
    model_path = "path_to_model"
    data_path = './data/dev-v1.1.json'
    with open(data_path, 'r') as f:
        dataset = json.load(f)['data']
    f.close()
    
    interval = len(dataset) // num_workers
    start_indices = list(range(0, len(dataset), interval))
    end_indices = start_indices[1:] + [len(dataset)]
    model_path_list, dataset_list, res_list, device_list, mode_list = [model_path] * num_workers, [dataset] * num_workers, [res] * num_workers, [_ for _ in range(num_workers)], ['zero-shot'] * num_workers
    
    # Do not calculate logprob, support one-shot and zero-shot
    process_map(generate_answer_finetune, model_path_list, dataset_list, res_list, start_indices, end_indices, device_list, mode_list, max_workers=num_workers)
    
    # Calculate logprob, use with explanation finetuned model, zero-shot only
    process_map(generate_answer_finetune_explanation, model_path_list, dataset_list, res_list, start_indices, end_indices, device_list, max_workers=num_workers)
    