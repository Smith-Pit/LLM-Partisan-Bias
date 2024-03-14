
from transformers import BertTokenizer, BertModel
import torch
from torch import nn



class Classifier(nn.Module):
  def __init__(self, n_classes):
    super(Classifier, self).__init__()
    self.bert = BertModel.from_pretrained("bert-base-uncased", return_dict=False)
    self.drop = nn.Dropout(p=0.3)
    self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

  def forward(self, input_ids, attention_mask):
    _, pooled_output = self.bert(
        input_ids = input_ids,
        attention_mask = attention_mask
    )

    output = self.drop(pooled_output)
    return self.out(output)


def infer_on_text(text, model):
  
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  
  tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

  encoded = tokenizer.encode_plus(text,
                                  max_length=512,
                                  add_special_tokens=True,
                                  return_token_type_ids=False,
                                  pad_to_max_length=True,
                                  return_attention_mask=True,
                                  truncation=True,
                                  return_tensors='pt')

  input_ids = encoded['input_ids'].to(device)
  attention_mask = encoded['attention_mask'].to(device)

  output = model(input_ids, attention_mask)



  _, preds = torch.max(output, dim=1)

  
  #print(f"Text Prompt: {text}")
  pred_label = preds[0].cpu().data.numpy()
  #print(f"Prediction: {pred_label} with type = {type(pred_label)}")
  return pred_label