import torch
from typing import Optional
from .eenet import EarlyExitNetwork


def test_all_exits_accuracy(model:EarlyExitNetwork, 
                            dataloader:torch.utils.data.DataLoader,
                            device: Optional[torch.device] = None):
    '''
    A function that tests the accuracy of ALL the exits in a model.
    pay close attention to the shapes of the outputs and the labels.
    For each sample and this each label, there are n outputs where n is the number of exits.
    '''
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    model = model.to(device)
    model.eval()
    with torch.no_grad():     
        # we take the first batch to get the number of exits, to infer the number of exits and the number of classes
        # without having the user provide them.
        images,_ = next(iter(dataloader)) # we ignore the labels
        images  = images.to(device)
        outputs=  model(images)
        num_exits = outputs.shape[1]
        correct = [0] * num_exits # we keep how many sample each exit got correclty
        total = 0 # total samples are the same for all exits
        for images,labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            batch_size =outputs.shape[0] # the batch size can change (if len(dataset) % batch_size !=0) so we check each time

            for b in range(batch_size): # we itterate over the batch, because we need to check each answer for each exit
                correct_answer = labels[b] # for all of the exits, the correct answer is the same
                for e in range(num_exits): # for each exit 
                    prediction = torch.argmax(outputs[b][e],dim=0) # we get the answer
                    if prediction.item() == correct_answer: 
                        correct[e] +=1 

            total += batch_size
        return correct,total
   