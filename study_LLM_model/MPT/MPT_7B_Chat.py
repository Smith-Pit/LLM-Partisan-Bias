#!pip install transformers, accelerate, einops, langchain, xformers
#Code from https://github.com/pinecone-io/examples/blob/master/learn/generation/llm-field-guide/mpt/mpt-7b-huggingface-langchain.ipynb

#rsync MPT_7B_Chat.py pitp@spartan.hpc.unimelb.edu.au:/data/gpfs/projects/punim0619/Smith/study_LLM_model/MPT

from torch import cuda, bfloat16
import transformers
import torch
from transformers import StoppingCriteria, StoppingCriteriaList
import sys
import pandas as pd

sys.path.insert(0, '/data/gpfs/projects/punim0619/Smith/model/helpers')
from helpers import infer_on_text, Classifier

# OCCUPATIONS = ["Economist", "Journalist", "Businessman", "Professor", "Military solider", "Government employee"]
# RACES = ["White American", "Asian American", "Black American", "Hispanic American"]

# IDEO_LEAK = "Republican Politician"

finance = ["Financial Advisor", "Loan Officer", "Bank Manager", "Auditor", "Investment Banker"]
healthcare = ["Doctor",  "Healthcare Administrator", "Pharmacist", "Hospital Clerk", "Insurance Provider"]
retail = ["Retail Store Manager", "Supply Chain Manager", "Marketing Officer", "Wholesaler", "Hiring Manager"]
education = ["Parent"]

all_occ = [education]

def init_model():
    model = Classifier(3)
    model.load_state_dict(torch.load('/data/gpfs/projects/punim0619/Smith/model/100.0_weight.pth'))
    model.eval()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    return model

device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'

model = transformers.AutoModelForCausalLM.from_pretrained(
    'mosaicml/mpt-7b-chat',
    trust_remote_code=True,
    torch_dtype=bfloat16,
    max_seq_len=2048
)
model.eval()
model.to(device)
print(f"Model loaded on {device}")

#in this module, there is already a var called model. That is the MPT model
classi_model = init_model()

tokenizer = transformers.AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")

# mtp-7b is trained to add "<|endoftext|>" at the end of generations. After generating this special token, it most likely will 
#hallucinate. 
stop_token_ids = tokenizer.convert_tokens_to_ids(["<|endoftext|>"])

# define custom stopping criteria object
class StopOnTokens(StoppingCriteria):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        for stop_id in stop_token_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False

#this stopping criteria will ensure that the model stops generating new tokens after the special token
stopping_criteria = StoppingCriteriaList([StopOnTokens()])


generate_text = transformers.pipeline(
    model=model, 
    tokenizer=tokenizer,
    return_full_text=True,  # langchain expects the full text
    task='text-generation',
    device=device,
    # we pass model parameters here too
    stopping_criteria=stopping_criteria,  # without this model will ramble
    temperature=0.1,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
    top_p=0.15,  # select from top tokens whose probability add up to 15%
    top_k=0,  # select from top 0 tokens (because zero, relies on top_p)
    max_new_tokens=512,  # mex number of tokens to generate in the output
    repetition_penalty=1.1  # without this output begins repeating
)


prompt_file = "/data/gpfs/projects/punim0619/Smith/data/prompts_VAR.txt"
input_file = open(prompt_file, 'r', encoding='latin-1')
input_lines = input_file.readlines()

for industry in all_occ:

    for occ in industry:
        data_dict = {}
        ##generate response
        for line in input_lines:

            line = line.replace("VAR", occ)

            prompt = f"<|im_start|>user\n{line}<|im_end|> <|im_start|>assistant"
            
            res = generate_text(prompt)

            component = res[0]['generated_text'].split("<|im_start|>")

            question = component[1].removeprefix("user\n")
            question = question.removesuffix("<|im_end|> ")

            response = component[2].removeprefix("assistant")

            
            new_string = "[CLS]" + question + "[SEP]" + response
            label = infer_on_text(new_string, classi_model)

            data_dict[question] = label

        df = pd.DataFrame(data_dict.items(), columns=['Question', 'Label'])
        race_ = occ.split(" ")[0]
        df.to_csv(f'/data/gpfs/projects/punim0619/Smith/data/outputs/MPT/MPT_{race_}_prompts.csv')




##

##generate Political Compass answers
# for line in input_lines:
#     prompt = line + "You must only answer with one of the following options: Strongly Agree, Agree, Disagree, Strongly Disagree."
#     res = generate_text(prompt)
#     print(res[0]["generated_text"])
#     print("\n==================================\n")
    
