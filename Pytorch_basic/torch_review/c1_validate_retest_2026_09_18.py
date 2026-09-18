import torch
from torch import nn

def validate(model,fn_loss,dataloader):
    with torch.no_grad():
        full_loss=0.0
        batch_size=0
        for batch_x,batch_y in dataloader:
            pred_y=model(batch_x)
            batch_loss=fn_loss(pred_y,batch_y)
            full_loss+=batch_loss*len(batch_x)
            batch_size+=len(batch_x)
    return full_loss/batch_size