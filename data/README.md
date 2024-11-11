### This directory stores data files for training and test. The  funtionalities of all files are listed here:
- dev-small.json: A small json file contains only one example from the develop(test) set
- dev-v1.1.json: The original file of SQuAD version 1.1 develop(test) data
- dev-v1.1_formatted.json: The formatted json file of dev-v1.1.json(indent=4) for easier reading
- train-v1.1.json: The original file of SQuAD version 1.1 training dataset
- train-v1.1_formatted.json: The formatted json file of train-v1.1.json(indent=4) for easier reading
- train-v1.1_with_explanation.json: This file stores the training data augmented with explanations. Explanations are generated via the GPT-4o-mini model API

Remeber to adjust the path in .py files if there is any inconsistency.