import torch
import torch.nn as nn
from copy import deepcopy


class EarlyExitNetwork(nn.Module):
    '''
    This is a class that taken A SEPARATED network and a modulelist of exit layers and creates an early exit network
    The separated network  is the original network that the user has manually separated and placed into a
    nn.Sequential container.
    Each exit will be placed at the end of each part of the sequential network.
    Dimension matching is left to the user.
    The last exit is the output if part of the original network so the exits provided have to be n-1 where n is the number of parts in the separated network.
    '''
    
    def __init__(self,
                 separated_network: nn.Sequential,
                 exit_layers: nn.ModuleList):
        super(EarlyExitNetwork, self).__init__()
        # if the network is broken up in n different pieces, the exit_layers should be  n-1
        # as the last exit is the output layers of the original model
        assert len(exit_layers) == len(separated_network)-1
        self.network = deepcopy(separated_network[:-1])
        self.exits = deepcopy(exit_layers)
        self.exits.append(deepcopy(separated_network[-1]))

        self.len = len(self.network)

    def forward(self,x,exit_chosen=None):
        '''
        When not providing an exit, all of them will be used simultaneously.
        This is mostly used if one want to train or test the whole network.
        '''
        if exit_chosen is None:
            outputs = []
            for i in range(self.len):
                # itterate over the core network
                x = self.network[i](x)
                
                # store the exit in the outputs
                early_exit = self.exits[i](x)
                outputs.append(early_exit)
            
            # the last exit is part of the original network
            x = self.exits[-1](x)
            outputs.append(x)
            # very carefull with shapes and dimensions, dim =0 is the batch so we concat on dim=1
            return torch.stack(outputs, dim=1)
        else:
            # if we want only one exit, make sure we ae between 0 and the number of exits
            assert 0<=exit_chosen<=self.len
            # if we chose a layer other than the last one, we go through the network
            # if we chose the last one, it does not have a network layer, as it is attached to the exit
            network_layers = min(exit_chosen,self.len-1)
            x = self.network[0](x)
            for i in range(1,network_layers+1):
                x = self.network[i](x)

            return self.exits[exit_chosen](x)
