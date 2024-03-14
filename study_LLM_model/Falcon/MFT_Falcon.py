#!nvidia-smi
from torch import cuda
from langchain import HuggingFacePipeline
from transformers import AutoTokenizer, pipeline
from langchain import PromptTemplate,  LLMChain
import pandas as pd
#from ctransformers import AutoModelForCausalLM
import sys
import torch
import json


model = "tiiuae/falcon-7b-instruct" #tiiuae/falcon-40b-instruct
device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'
tokenizer = AutoTokenizer.from_pretrained(model)

pipeline = pipeline(
    "text-generation", #task
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto",
    max_length=2048,
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id
)

llm = HuggingFacePipeline(pipeline = pipeline, model_kwargs = {'temperature':0})

template = """
You are an intelligent chatbot. Help the following question with brilliant answers.
Question: {question}
Answer:"""
prompt = PromptTemplate(template=template, input_variables=["question"])
llm_chain = LLMChain(prompt=prompt, llm=llm)

prompt_file = "/data/gpfs/projects/punim0619/Smith/data/MFT_Prompt.txt"
input_file = open(prompt_file, 'r', encoding='latin-1')
input_lines = input_file.readlines()

data_dict = {}

for line in input_lines:

    question = line
    response = llm_chain.run(question)

    print(question)
    print('\n')
    print(response)
    print('\n=======================================\n')

    data_dict[question] = response


df = pd.DataFrame(data_dict.items(), columns=['Question', 'Response'])
df.to_csv('/data/gpfs/projects/punim0619/Smith/data/outputs/Falcon/Falcon_MFT.csv')

    