import openai
import json, os
from tqdm import tqdm

with open('../data/train-v1.1_formatted.json', 'r') as file:
	data = json.load(file)

openai_client = openai.Client(
	api_key=os.environ['OPENAI_API_KEY']
)

assistant_prompt = (
	'The user is going to give you a passage, a relevant question and the answer to the question. '
	'The answer to the question is in the passage. You are going to give the reason why the answer fits the question and where you find the answer. '
	'Please output in the following format strictly and give your reason briefly:\n'
	'Answer: # [The answer to the question] #\n'
	'Reason: [The reason]\n'
)

final_res = []

for d in tqdm(data[:10], desc="Data"):
	for p in tqdm(d['paragraphs'], desc="Paragraphs"):
		for q in tqdm(p['qas'], desc="Questions"):
			messages = [
				{'role': 'assistant', 'content': assistant_prompt},
				{'role': 'user', 'content': 'Passage: ' + p['context'] + '\nQuestion: ' + q['question'] + '\nAnswer: ' +
											q['answers'][0]['text']}
			]
			response = openai_client.chat.completions.create(
				model='gpt-4o-mini',
				messages=messages,
			)

			res = response.choices[0].message.content
			item = {
				'id': q['id'],
				'title': d['title'],
				'context': p['context'],
				'question': q['question'],
				'answers': q['answers'][0]['text'],
				'explanation': res
			}
			final_res.append(item)
		with open('processed_data.json', 'w') as f:
			json.dump(final_res, f, ensure_ascii=False, indent=4)
