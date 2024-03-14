# -*- coding: utf-8 -*-
"""FALCON with LangChain

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1X1ulWEQxE7MNV6qxrmEpXtLZ1nlVYUDQ
"""

#!pip install -q transformers einops accelerate langchain bitsandbytes

#!nvidia-smi

from langchain import HuggingFacePipeline
from transformers import AutoTokenizer, pipeline
#from ctransformers import AutoModelForCausalLM
import sys
import torch

model = "tiiuae/falcon-7b-instruct" #tiiuae/falcon-40b-instruct

tokenizer = AutoTokenizer.from_pretrained(model)


pipeline = pipeline(
    "text-generation", #task
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto",
    max_length=200,
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id
)

llm = HuggingFacePipeline(pipeline = pipeline, model_kwargs = {'temperature':0})

from langchain import PromptTemplate,  LLMChain

template = """
You are an intelligent chatbot. Help the following question with brilliant answers like how a Republican politician would.
Question: {question}
Answer:"""
prompt = PromptTemplate(template=template, input_variables=["question"])

file = open(sys.argv[1], 'r')
lines = file.readlines()

llm_chain = LLMChain(prompt=prompt, llm=llm)

##writing into txt file
index = 0
#roles = ["a university student","a small business owner","a journalist", "a highschool teacher","a university professor","the President of the United states","an AI assistant"]
with open(sys.argv[2], mode="w") as output_file:
    for line in lines:
        #for role in roles:
        #question = line.replace("_VAR", role)
        question = line
        #output_file.write(f"Question #{index} (Role:{role}): {question}\n")
        output_file.write(f"Question #{index}: {question}\n")
        #output_file.write(f"Answer #{index} (Role:{role}) : {llm_chain.run(question)}\n")
        output_file.write(f"Answer #{index}: {llm_chain.run(question)}\n")
        output_file.write('\n')

    index += 1