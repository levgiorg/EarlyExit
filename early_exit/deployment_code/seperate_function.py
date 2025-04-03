from .seperated_model import SegmentedEarlyExitNetwork
import torch
import torch.nn as nn


def seperate_networks(eenet,thresholds,confidence_function=None,device='cpu'):
    segmented_networks =nn.ModuleList([])
    for i in range((len(eenet.network))):
        network = SegmentedEarlyExitNetwork(eenet.network[i],eenet.exits[i],thresholds[i],confidence_function).to(device)
        segmented_networks.append(network)
    network  = SegmentedEarlyExitNetwork(eenet.exits[-1]).to(device)
    segmented_networks.append(network)
    return segmented_networks

def test_networks(networks,dataset,device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    correct = 0
    total = len(dataset)
    number_of_exits_taken = {}
    for i in range(len(networks)):
        number_of_exits_taken[i] = {'correct':0,'total':0}
    with torch.no_grad():
        for images, labels in dataset:
            tensor = images.to(device)
            
            exit_taken = False
            model_layer = 0

            while not exit_taken:
                tensor,exit_taken = networks[model_layer](tensor)
                model_layer +=1
                
            number_of_exits_taken[model_layer-1]['total'] +=1
            if torch.argmax(tensor) == labels:
                correct +=1
                number_of_exits_taken[model_layer-1]['correct'] +=1
                
    return correct / total,number_of_exits_taken
