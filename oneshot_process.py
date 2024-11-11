import json, os, re

def process_data(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
        f.close()
    
    final_res = {}
    for qid, item in data.items():
        match = re.search(r':', item, re.IGNORECASE)
        if match:
            res = item[match.end():].strip()
        else:
            match = re.search(r'answer', item, re.IGNORECASE)
            if match:
                res = item[match.end():].strip()
            else:
                res = item
        final_res[qid] = res
        
    base, ext = os.path.splitext(file_path)
    new_file_path = f"{base}_processed{ext}"
    
    with open(new_file_path, 'w') as f:
        json.dump(final_res, f, ensure_ascii=False, indent=4)
    
    print(f"Processed data has been saved to {new_file_path}")
    
    

if __name__ == '__main__':
    file_path = './result/phi-3.5-mini-instruct-one-shot.json'
    # file_path = './result/llama-3.2-3B-instruct-one-shot.json'
    process_data(file_path)