import json
import random
import re

with open('./result/finetune_explanation/llama-3.2-3b-explanation-tuned.json', 'r') as f:
    data_llama = json.load(f)
    f.close()
    
with open('./result/finetune_explanation/phi-3.5-mini-instruct-explanation-tuned.json', 'r') as f:
    data_phi = json.load(f)
    f.close()
    
final_data = {}

for qid, item in data_llama.items():
    score_llama = data_llama[qid]['prob']
    score_phi = data_phi[qid]['prob']
    
    same_cnt = 0
    if score_llama > score_phi:
        final_data[qid] = data_llama[qid]['res']
    elif score_phi > score_llama:
        final_data[qid] = data_phi[qid]['res']
    else:
        same_cnt += 1
        final_data[qid] = random.choice([data_llama[qid]['res'], data_phi[qid]['res']])

print(same_cnt)
            
for qid, item in final_data.items():
    pattern = r"#(.*?)#"

    match = re.search(pattern, item)
    if match:
        final_data[qid] = match.group(1).strip()
    else:
        final_data[qid] = item
        
with open('./result/finetune_explanation/final.json', 'w') as f:
    json.dump(final_data, f, indent=4)
    f.close()