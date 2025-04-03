from tqdm import tqdm
import torch
import torch.nn as nn
from typing import Optional
from .eenet import EarlyExitNetwork


def train_model(model:EarlyExitNetwork,
                loader: torch.utils.data.DataLoader,
                criterion: nn.Module,
                optimizer: torch.optim.Optimizer, 
                epochs:int=1,
                exit_chosen:Optional[int] = None,
                device:Optional[torch.device] = None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # progress bar
    loop = tqdm(loader)
    # we take the first batch to get the number of exits, to infer the number of exits and the number of classes
    # without having the user provide them.
    images,_ = next(iter(loader))
    images = images.to(device)
    output = model(images)
    
    # if we dont chose an exit, we need to know how many exits we have
    if exit_chosen is None:
        num_exits = output.shape[1]
    num_classes = output.shape[-1]
    for epoch in range(epochs):
        for images,labels in loop:
            images= images.to(device)
            
            labels = torch.nn.functional.one_hot(labels, num_classes=num_classes).float().to(device)
            output = model(images,exit_chosen=exit_chosen)
            optimizer.zero_grad()

            if exit_chosen is None:
                loss = 0 
                for i in range(num_exits):
                    loss += criterion(output[:,i,:],labels)
            else:
                loss = criterion(output,labels)
            loss.backward()
            optimizer.step()
            loop.set_postfix(
                loss=loss.item()
            )