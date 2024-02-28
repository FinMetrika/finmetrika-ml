import torch
from datasets import DatasetDict
from transformers import PreTrainedTokenizerBase



def fwd_pass(data_sample:DatasetDict,
             model,
             device:str,
             tokenizer:PreTrainedTokenizerBase):
    
    #predictions = []
    
    inputs = {
            k:v.to(device) for k,v in data_sample.items() if k in tokenizer.model_input_names
        }
    
    with torch.no_grad():
        output = model(**inputs)

        pred_label = torch.argmax(output.logits, axis=-1)
        #predictions.extend(pred_label)
    
    return {"predicted_label": pred_label.cpu().numpy()}



def accuracy_metrics():
    