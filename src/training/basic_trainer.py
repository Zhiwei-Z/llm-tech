from dataclasses import dataclass
import torch
from torch import nn
from torch import Tensor

from device import DEVICE

PAD_TOKEN_ID=0

def train(model: nn.Module,
          train_loader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          num_epochs: int,
          loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN_ID),
          ):

    # TODO: add eval code
    for epoch in range(num_epochs):
        with model.train():
            for batch_idx, (x, y) in enumerate(train_loader):
                # x, y are both tensors
                x, y = x.to(DEVICE), y.to(DEVICE)

                optimizer.zero_grad()
                logits = model(x)

                loss = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))

                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                loss.backward()
                optimizer.step()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")
