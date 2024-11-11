"""
zero_shot_Llama-3.2-3B-Instruct
"""

import torch
from transformers import pipeline
import json
from tqdm import tqdm
device = "cuda"
llama = "meta-llama/Llama-3.2-3B-Instruct"
model = pipeline(model=llama, device=device, torch_dtype=torch.bfloat16)

def ask_model(context, question):
    qa_prompt = f"{context}\n  {question}\n please ONLY give the direct answer itself without saving any other information including introductory remarks prompts and any periods. Remember, the answer must exist within the context provided. Extract it without adding any extraneous information.\n"

    prompt = [
        {"role": "system", "content": "You are a helpful assistant. Your task is to extract the answer directly from the provided context."},
        {"role": "user", "content": qa_prompt},
    ]

    generation = model(
        prompt,
        do_sample=False,
        temperature=1.0,
        top_p=1,
        max_new_tokens=50
    )

    answer = generation[0]['generated_text'][-1]['content']

    return answer

def load_squad_data(filepath):
    """
    Reads a dataset in SQuAD format from a specified file path.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        squad_data = json.load(f)
    return squad_data


def generate_predictions(dev_data, output_file):
    """
    Iterate through each question in the test dataset, invoke the model for inference, generate answers, and save predictions.
    """
    predictions = {}
    total_questions = sum(len(paragraph['qas']) for data in dev_data['data'] for paragraph in data['paragraphs'])


    with tqdm(total=total_questions, desc="Processing Questions") as pbar:
      for data in dev_data['data']:  
          for paragraph in data['paragraphs']:
              context = paragraph['context']  
              for qa in paragraph['qas']:
                  question = qa['question']  
                  question_id = qa['id']     

                  answer = ask_model(context, question)
                  predictions[question_id] = answer
                #   print(question)
                #   print(f"Generated Answer: {answer}")
                  pbar.update(1)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(predictions, f, ensure_ascii=False, indent=4)


def main():
    dev_data_filepath = "dev-v1.1.json"
    dev_data = load_squad_data(dev_data_filepath)

    output_filepath = "llama-3.2-3b-base.json"
    generate_predictions(dev_data, output_filepath)

if __name__ == "__main__":
    main()
