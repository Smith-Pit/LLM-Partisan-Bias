
from torch import cuda, bfloat16
import transformers
import torch
from transformers import StoppingCriteria, StoppingCriteriaList
import sys
import pandas as pd

device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'

name = 'mosaicml/mpt-7b-instruct'
config = transformers.AutoConfig.from_pretrained(name, trust_remote_code=True)
config.max_seq_len = 4096 # (input + output) tokens can now be up to 4096

model = transformers.AutoModelForCausalLM.from_pretrained(
    name,
    trust_remote_code=True,
    config=config,
    torch_dtype=bfloat16
)
model.eval()
model.to(device)
print(f"Model loaded on {device}")

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

prompt_file = "/data/gpfs/projects/punim0619/Smith/data/outputs/MPT/prompts.csv"
input_df = pd.read_csv(prompt_file)

data_dict = {}


##generate response
for i in range(len(input_df)):

    question_pair = input_df.iloc[i]["Question"]
    component = question_pair.split("[SEP]")
    question = component[0].removeprefix("[CLS]")
    response = component[1]

    sentence = f"With the question = {question} and the response = {response}. Do you think the response is politically liberal, politcally conservative or neutral?"
    
    prompt = f"<|im_start|>user\n{sentence}<|im_end|> <|im_start|>assistant"
    res = generate_text(prompt)

    model_response = res[0]['generated_text'].split("<|im_start|>")[-1]

    data_dict[i] = model_response
  
    print(model_response)


df = pd.DataFrame(data_dict.items(), columns=['Question_Index', 'Perceived_Label'])
df.to_csv('/data/gpfs/projects/punim0619/Smith/data/outputs/MPT/MPT_SelfPerception.csv')




