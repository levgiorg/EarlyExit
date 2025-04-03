# TARDIS Early Exit

A PyTorch framework for converting standard neural networks into early exit models to improve computational efficiency and inference speed.

## Overview

Early exit networks allow inference to terminate at intermediate layers when a confident prediction is made, reducing computational overhead for "easy" inputs while maintaining full network capabilities for more complex inputs. This framework provides tools to:

1. Convert existing PyTorch models into early exit networks
2. Train and fine-tune early exit models
3. Deploy segmented early exit networks with confidence thresholds
4. Test and evaluate early exit performance

## Dependencies

The following dependencies are required:
- Python 3.6+
- PyTorch 2.0+
- torchvision (for demos using image datasets)
- tqdm (for training progress visualization)
- numpy
- filelock
- fsspec
- Jinja2
- MarkupSafe
- mpmath
- networkx
- sympy
- typing_extensions

You can install all dependencies using:
```
pip install -r requirements.txt
```

## Installation

```bash
# Clone the repository
git clone https://github.com/username/TARDIS-EarlyExit.git

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

See `demo.ipynb` for a complete demonstration using VGG19 on CIFAR-10.

### Basic Usage

1. **Import the necessary modules:**
   ```python
   from early_exit import EarlyExitNetwork, train_model, test_all_exits_accuracy
   from early_exit import seperate_networks, SegmentedEarlyExitNetwork, test_networks
   import torch.nn as nn
   import torch
   ```

2. **Split your model into sequential blocks:**
   ```python
   # Example with a VGG model
   device_side_net = deepcopy(model.features[:7])
   server_side_net = deepcopy(model.features[7:17])
   cloud_side_net = nn.Sequential(
       deepcopy(model.features[17:]),
       nn.Flatten(),
       deepcopy(model.classifier)
   )
   
   core_net = nn.Sequential(device_side_net, server_side_net, cloud_side_net)
   ```

3. **Define early exit layers:**
   ```python
   # Create exit layers based on intermediate feature shapes
   exits = nn.ModuleList([
       nn.Sequential(
           nn.Flatten(),
           nn.Linear(intermediate_shape_1, 512),
           nn.Linear(512, num_classes)
       ),
       nn.Sequential(
           nn.Flatten(),
           nn.Linear(intermediate_shape_2, 512),
           nn.Linear(512, num_classes)
       )
   ])
   ```

4. **Create the early exit network:**
   ```python
   eenet = EarlyExitNetwork(core_net, exits).to(device)
   ```

5. **Train the network:**
   ```python
   # Train all exits simultaneously
   optimizer = torch.optim.Adam(eenet.parameters(), lr=1e-4)
   criterion = nn.BCEWithLogitsLoss()
   train_model(eenet, trainloader, criterion, optimizer, epochs=5)
   
   # Or train a specific exit
   exit_to_train = 0
   optimizer = torch.optim.Adam(eenet.exits[exit_to_train].parameters(), lr=1e-4)
   train_model(eenet, trainloader, criterion, optimizer, exit_chosen=exit_to_train)
   ```

6. **Test the network:**
   ```python
   # Test all exits simultaneously
   correct, total = test_all_exits_accuracy(eenet, testloader)
   for i, c in enumerate(correct):
       print(f"Exit {i} accuracy: {c/total:.4f}")
   ```

7. **Deploy as segmented networks:**
   ```python
   # Create segmented networks with confidence thresholds
   networks = seperate_networks(eenet, thresholds=[0.9, 0.9])
   
   # Test segmented networks
   accuracy, exit_stats = test_networks(networks, testset)
   print(f"Overall accuracy: {accuracy:.4f}")
   print("Exit statistics:", exit_stats)
   
   # Save segmented models
   for i, net in enumerate(networks):
       torch.save(net, f"model_{i}.pth")
   ```

## Detailed Implementation Guide

### 1. Prepare Your Model

Convert your existing model into early exit architecture by:

- **Step 1**: Analyze your model to determine appropriate exit points
- **Step 2**: Split the model into sequential blocks
- **Step 3**: Design exit layers for each intermediate point

### 2. Creating an Early Exit Network

The `EarlyExitNetwork` class combines:
- Your sequential sub-networks (blocks of the original model)
- Exit layers at intermediate points

The last exit is automatically set to the final component of your original network.

```python
# Structure:
# - self.network: List of network segments
# - self.exits: List of exit layers
```

### 3. Training Options

- **Full network training**: Trains all exits simultaneously
- **Single exit training**: Targets specific exits for fine-tuning
- **Custom loss weighting**: Customize importance of each exit

### 4. Deployment

The framework provides two deployment options:

1. **Single Network with Exit Selection**: Use the original `EarlyExitNetwork` and specify `exit_chosen` during inference
2. **Segmented Deployment**: Use `seperate_networks` to create independent networks with confidence thresholds

### 5. Confidence-Based Early Exiting

The `SegmentedEarlyExitNetwork` class supports:
- Custom confidence functions (default: Softmax)
- Adjustable confidence thresholds
- Automatic progression to deeper exits when confidence is below threshold

## Performance Considerations

- Early exits typically trade some accuracy for computational efficiency
- Tuning confidence thresholds balances accuracy vs. computational savings
- Different exit architectures may be needed based on feature map dimensions
- Consider different training strategies for different deployment scenarios

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Feedback & Issues

Please submit issues and feature requests through the project's issue tracker.

## Citation

If you use this framework in your research, please cite:

```
[Citation information]
```

## Acknowledgment

This work was partially supported by the "Trustworthy And Resilient Decentralised Intelligence For Edge Systems (TaRDIS)" Project, funded by EU HORIZON EUROPE program, under grant agreement No 101093006.


