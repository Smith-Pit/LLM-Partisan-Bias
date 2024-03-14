
#rsync Flan_UL2.py pitp@spartan.hpc.unimelb.edu.au:/data/gpfs/projects/punim0619/Smith/study_LLM_model/Flan_UL2

from transformers import T5Tokenizer, T5ForConditionalGeneration
import pandas as pd
import sys
import torch

sys.path.insert(0, '/data/gpfs/projects/punim0619/Smith/model/helpers')
from helpers import infer_on_text, Classifier


tokenizer = T5Tokenizer.from_pretrained("google/flan-ul2")
model = T5ForConditionalGeneration.from_pretrained("google/flan-ul2", device_map="auto", load_in_8bit=True)

# OCCUPATIONS = ["Economist", "Journalist", "Businessman", "Professor", "Military solider", "Government employee"]
# RACES = ["White American", "Asian American", "Black American", "Hispanic American"]

finance = ["Financial Advisor", "Loan Officer", "Bank Manager", "Auditor", "Investment Banker"]
healthcare = ["Doctor",  "Healthcare Administrator", "Pharmacist", "Hospital Clerk", "Insurance Provider"]
retail = ["Retail Store Manager", "Supply Chain Manager", "Marketing Officer", "Wholesaler", "Hiring Manager"]
education = ["University Professor", "Student", "School  Administrator", "Textbook Publisher", "Parent"]

all_occ = [finance, healthcare, retail, education]

def init_model():
    model = Classifier(3)
    model.load_state_dict(torch.load('/data/gpfs/projects/punim0619/Smith/model/100.0_weight.pth'))
    model.eval()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    return model

prompt_file = "/data/gpfs/projects/punim0619/Smith/data/prompts_VAR.txt"
input_file = open(prompt_file, 'r', encoding='latin-1')
input_lines = input_file.readlines()

DIRECT = "Based of the given statement, choose one of the following options: Strongly Agree, Agree, Disagree, Strongly Disagree."
IDEO_LEAK = "Republican Politician"

classi_model = init_model()


for industry in all_occ:
    for occ in industry:

        data_dict = {}

        for line in input_lines:

            #prompt = line
            prompt = line.replace("VAR", occ)
            #prompt = line + DIRECT
    
            inputs = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
            outputs = model.generate(inputs, max_length=200)
            response = tokenizer.decode(outputs[0])

            new_string = "[CLS]" + prompt + "[SEP]" + response

            label = infer_on_text(new_string, classi_model)

            data_dict[prompt] = label

        df = pd.DataFrame(data_dict.items(), columns=['Question', 'Label'])
        race_ = occ.split(" ")[0]
        df.to_csv(f'/data/gpfs/projects/punim0619/Smith/data/outputs/Flan_UL2/Flan_UL2_{race_}_prompts.csv')
                



