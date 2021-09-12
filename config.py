import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # sets device for model and PyTorch tensors
print_freq = 400  # print training/validation stats  every __ batches
