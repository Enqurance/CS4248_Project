# llama finetune without explanation
def format_chat_template(row):
    instruction = "You are a helpful assistant. Your task is to extract the answer directly the provided context."
    context = row['context']
    question = row['question']
    user_prompt = f"{context}\n {question}\n please ONLY give the direct answer itself without saving any other information including introductory remarks prompts and any periods. Remember, the answer must exist within the context provided. Extract it without adding any extraneous information.\n"
    
    row_json = [
        {"role": "system", "content": instruction},
        {"role": "user", "content": user_prompt},
        {"role": "system", "content": row["answers"]["text"][0]}
    ]

    row["text"] = tokenizer.apply_chat_template(row_json, tokenize=False)
    return row

# llama finetune with explanation
def format_chat_template(row):
    context = row['context']
    question = row['question']
    explanation = data_exp_dict[row['id']]["explanation"]
    user_prompt = f"{context}\n {question}\n please ONLY give the direct answer itself without saving any other information including introductory remarks prompts and any periods. Remember, the answer must exist within the context provided. Extract it without adding any extraneous information.\n"
    
    row_json = [
        {"role": "system", "content": instruction},
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": explanation}
    ]

    row["text"] = tokenizer.apply_chat_template(row_json, tokenize=False)
    return row

# phi finetune without explanation
def format_chat_template(row):
    instruction = "You are a helpful assistant. Your task is to extract the answer directly the provided context."
    context = row['context']
    question = row['question']
    user_prompt = f"{context}\n {question}\n please ONLY give the direct answer itself without saving any other information including introductory remarks prompts and any periods. Remember, the answer must exist within the context provided. Extract it without adding any extraneous information.\n"
    
    row_json = [
        {"role": "assistant", "content": instruction},
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": row["answers"]["text"][0]}
    ]

    row["text"] = tokenizer.apply_chat_template(row_json, tokenize=False)
    return row

# phi finetune with explanation
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