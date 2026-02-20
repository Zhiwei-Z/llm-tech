from dataclasses import dataclass

@dataclass
class SharedConfig:
  d_model: int
  vocab_size: int
