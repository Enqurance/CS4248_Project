# This is the final project for CS4248, group 15

## Quick Start

You can setup the environment for running this repo quickly with conda. First you need to create the environment.

```
conda env create -f environment.yml
```

Then activate it.

```
conda activate cs4248_project
```

## Finetune
**The Funtune step is run on Nvidia 3090 GPU**

You may use the file `finetune.py` to tune the model with proposed conversation templates. Before doing that, you need to export your huggingface token and wandb token in your enviroment. You may visit [HuggingFace](https://huggingface.co) and [wandb](https://wandb.ai) for more information. To configure your conversation template, please modify the function `format_chat_template`:

```
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
```

Please note that different LLMs support various conversation templates, such as different roles and language styles. When using a fine-tuned model, it is important to apply the conversation template that was used during the model's tuning process. Please adjust the conversation template when doing inference. For this repo, you can find the `format_chat_template` we used for differen tasks in file `tune_templates.py` and you can directly copy&paste.

When tuning, you need to specify the path of the model and the path where the Adapter is saved:

```
base_model = "model_path"
new_model = "adapter_save_path"
```

Then you can start tuning:
```
python finetune.py
```

After tuning, you can view the result on wandb. You need to merge the Adapter with the original model to get your tuned LLM. This is done by `merge.py`. You need to designate the Adapter's path and original model's path:

```
base_model = "model_path"
new_model = "adapter_save_path/checkpoint-..."
save_path = "tuned_model_save_path"
```

Then just run the python script.

```
python merge.py
```

Now the finetune is finished.

## Inference
**The Inference step is run on Nvidia 3090 GPU**

You can use file `phi_inference.py` and `llama_inference.py` to run your finetuned model. This two files support zero-shot, one-shot and logprob calculation. Remember to configure the model paths to point to your specific model. Other configurations can be found in `main`.

Then just run the python script.

```
python phi_inference.py
python llama_inference.py
```

You may need some postprocessing after getting the result file. For example, extract answers from model's output. The next section will introduce how to evaluate using the experimental results we provided and they are already well processed.

## Evaluation

You may need to first process results after inferencing. For zero-shot and finetune withour explanation, you do not need to process result files. Simply running the `evaluate-v2.0.py` can do well.

```
python evaluate-v2.0.py ./data/dev-v1.1.json ./result/finetune_explanation/phi-3.5-mini-instruct-base.json
```

For one-shot, you can use the file `oneshot_process.py` to process. You can see details of how we do answer extratcion in this file. For finetuning with explanation, run file `select_scores.py` to get results by logprob ensemble.

Please pay attention that preferred format for evaluation script is:
```
{
    '<qid>': '<result>',
    ...
}
```

We have already provided all our experimental results under `./result` directory.