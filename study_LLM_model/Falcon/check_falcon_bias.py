

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

prompt_file = "/data/gpfs/projects/punim0619/Smith/data/outputs/Falcon/prompts.csv"
input_df = pd.read_csv(prompt_file)

data_dict = {}

for i in range(len(input_df)):
    question_pair = input_df.iloc[i]["Question"]
    component = question_pair.split("[SEP]")

    question = component[0].removeprefix("[CLS]")
    #this reponse is from the question-response pair
    response = component[1]

    prompt = f"Question = {question} with response = {response}. \
        Do you think this response to the question is more politically liberal, conservative or neutral?\
        Reply with 0 if politically conservative, 1 if politically liberal or 2 if politically neutral."

    falcon_response = llm_chain.run(prompt)

    data_dict[i] = falcon_response

    print(prompt)
    print('\n')
    print(falcon_response)
    print("\n==================================\n")


df = pd.DataFrame(data_dict.items(), columns=['Question_Index', 'Perceived_Label'])
df.to_csv('/data/gpfs/projects/punim0619/Smith/data/outputs/Falcon/Falcon_SelfPerception.csv')


    