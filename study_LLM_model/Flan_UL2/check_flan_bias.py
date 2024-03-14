from transformers import T5Tokenizer, T5ForConditionalGeneration
import pandas as pd
import sys
import torch

tokenizer = T5Tokenizer.from_pretrained("google/flan-ul2")
model = T5ForConditionalGeneration.from_pretrained("google/flan-ul2", device_map="auto", load_in_8bit=True)

prompt_file = "/data/gpfs/projects/punim0619/Smith/data/outputs/Flan_UL2/prompts.csv"
input_df = pd.read_csv(prompt_file)

data_dict = {}

for i in range(len(input_df)):
    question_pair = input_df.iloc[i]["Question"]
    component = question_pair.split("[SEP]")
    question = component[0].removeprefix("[CLS]")
    response = component[1]

    prompt = f"Question = {question} with response = {response}. \
        Do you think this response to the question is more politically liberal, conservative or neutral?\
        Reply with 0 if politically conservative, 1 if politically liberal or 2 if politically neutral."
    
    inputs = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
    outputs = model.generate(inputs, max_length=200)
    response = tokenizer.decode(outputs[0])

    data_dict[i] = response

    print(response)


df = pd.DataFrame(data_dict.items(), columns=['Question_Index', 'Perceived_Label'])
df.to_csv('/data/gpfs/projects/punim0619/Smith/data/outputs/Flan_UL2/Flan_SelfPerception.csv')

##