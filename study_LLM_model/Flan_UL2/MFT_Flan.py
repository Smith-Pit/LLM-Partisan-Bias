from transformers import T5Tokenizer, T5ForConditionalGeneration
import pandas as pd
import sys
import torch

tokenizer = T5Tokenizer.from_pretrained("google/flan-ul2")
model = T5ForConditionalGeneration.from_pretrained("google/flan-ul2", device_map="auto", load_in_8bit=True)

prompt_file = "/data/gpfs/projects/punim0619/Smith/data/MFT_Prompt.txt"
input_file = open(prompt_file, 'r', encoding='latin-1')
input_lines = input_file.readlines()

data_dict = {}

for line in input_lines:
    inputs = tokenizer(line, return_tensors="pt").input_ids.to("cuda")
    outputs = model.generate(inputs, max_length=200)
    response = tokenizer.decode(outputs[0])

    data_dict[line] = response
    print(response)


df = pd.DataFrame(data_dict.items(), columns=['Question', 'Response'])
df.to_csv('/data/gpfs/projects/punim0619/Smith/data/outputs/Flan_UL2/Flan_MFT.csv')
