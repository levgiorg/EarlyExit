import torch.nn as nn
import torch
from copy import deepcopy

class SegmentedEarlyExitNetwork(nn.Module):
        def __init__(self,network,exit=None,threshold=0.9,confidence_function=None):
            super(SegmentedEarlyExitNetwork, self).__init__()
            self.network = deepcopy(network)
            self.exit = deepcopy(exit)
            self.threshold = threshold
            if confidence_function is None:
                self.confidence_function=nn.Softmax(dim=0)
                
        def forward(self,x):
            # no batched input
            x = x.unsqueeze(0)
            x = self.network(x)

            if self.exit != None:
                early_exit = self.exit(x)

                early_exit=  early_exit.squeeze(0)
                confidence = self.confidence_function(early_exit)
                if confidence.max() > self.threshold:
                    return early_exit,True
                
                return x.squeeze(0),False
            
            x = x.squeeze(0)
            return x,True
