from parrot import Parrot
import torch
import warnings
import pandas as pd
from sklearn import preprocessing
from random import randint
warnings.filterwarnings("ignore")


#Init models (make sure you init ONLY once if you integrate this to your code)

parrot = Parrot(model_tag="prithivida/parrot_paraphraser_on_T5")

PATH = '/data/gpfs/projects/punim0619/Smith/model'

def paraphrase_text(text, model):

  para_phrases = parrot.augment(input_phrase=text,
                                use_gpu=True,
                                max_length = 512)


  while para_phrases == None:
    para_phrases = parrot.augment(input_phrase=text,
                                  use_gpu=True,
                                  max_length = 512)


  return para_phrases[0][0]



df = pd.read_csv(f"{PATH}/output.csv")
df.drop(columns=['Unnamed: 0'], inplace=True)
print('printing df\n')
print(df.head(5))


df_con = df[df['Label']==0]
df_con.reset_index(inplace=True, drop=True)

df_neu = df[df['Label']==2]
df_neu.reset_index(inplace=True, drop=True)

df_lib = df[df['Label'] == 1]

def increase_dataset(df, num_text):
  picked = set()

  label = df.iloc[0]["Label"]

  print(f"testing label = {label}")

  total_text = len(df) - 1

  for i in range(num_text):

    print(f"{i}/{num_text}")

    rand = randint(0, total_text)

    while (rand in picked):

      rand = randint(0, total_text)

    picked.add(rand)

    question_pair = df.iloc[rand]["Question"]

    components = question_pair.split("[SEP]")

    question = components[0].removeprefix("[CLS]")
    response = components[1]

    para_question = paraphrase_text(question, parrot)
    print('finished question para \n')
    para_response = paraphrase_text(response, parrot)

    new = "[CLS]" + para_question + "[SEP]" + para_response
    item = {"Question":new, "Label":label}
    df = df.append(item, ignore_index = True)


  return df

df_neu = increase_dataset(df_neu, 100)
df_neu.to_csv(f'{PATH}/df_neu.csv', index=False)

