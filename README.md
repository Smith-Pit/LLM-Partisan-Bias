# LLM-Partisan-Bias
This repository contains code for replicating the main findings in "Whose Side Are You On? Investigating the Political Stance of Large Language Models"

## Building BERTPOL
To train the classifier BERTPOL, the file classification_model.py in the "model" directory should be used. Datasets provided in the "model" directory include a balanaced dataset of liberal-conservative split ("balanced.csv") and a completely balanced dataset between liberal-conservative-neutral ("all-balanced.csv"). 

## Querying the models
To obtain responses from the models in this study, all of the architecture of each ones are located in the "study_LLM_model" directory. 

## 1. PEW Research Baseline result
For this study, the responses are collected by running the "model_name".py in the "study_LLM_model" directory against the "PEW_Research.txt" prompt set in the "data" directory. For example, Falcon-7B-Chat responses were collected from running study_LLM_model/Falcon/Falcon_7B.py against data/PEW_Research.txt

## 2. Indirect vs Direct Partisan Bias
For this study, the responses are collected by running the "model_name".py in the "study_LLM_model" directory against the "prompts_VAR.txt" prompt set in the "data" directory. The VAR should be replaced by "Democrat Politician" and "Republican Politician". The collected data for this is stored at data/"model_name"/democrats_prompt.csv or repub_prompts.csv

## 3. Occupational Role Political Stance
In this section, the responses were collected similarly to Section 2 above but the VAR is replaced by an occupation. The collected results from this study are stored at data/outputs/"model_name"/occ_"industry", for example Llama-2-7B-Chat's perception of a typical Loan Officer is stored at data/outputs/llama/occ_finance/llama_loan_prompts.csv

## 4. Self-Perception Check
For this study, the script in study_LLM_model/"model_name"/check_"model_name"_bias.py is used. 
