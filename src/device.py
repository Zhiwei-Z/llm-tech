import torch

'''
Set up torch device
'''

DEVICE = torch.device("cpu")
if torch.cuda.is_available():
  DEVICE = torch.device("cuda")
print(f"Torch using device {DEVICE}")