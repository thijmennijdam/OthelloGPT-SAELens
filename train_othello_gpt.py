from transformer_lens import HookedTransformer, HookedTransformerConfig
from transformer_lens.train import train, HookedTransformerTrainConfig
from datasets import load_dataset
import numpy as np
import torch as t
import os

device = t.device("cuda" if t.cuda.is_available() else "cpu")


model_cfg = HookedTransformerConfig(
    n_layers = 6,
    d_model = 128,
    d_head = 64,
    n_heads = 8,
    d_mlp = 128*4,
    d_vocab = 61,
    n_ctx = 59,
    act_fn="gelu",
    normalization_type="LN",
    device=device,
)

model = HookedTransformer(model_cfg).to(device)

train_cfg = HookedTransformerTrainConfig(
    lr=1e-4,
    batch_size=512,
    num_epochs=1,
    device=device,
    wandb=True,
    wandb_project_name="OthelloGPTTraining",
)

dataset = load_dataset("taufeeque/othellogpt", split="train")[:, :-1]


train(model, train_cfg, dataset)
t.save(model.state_dict(), f"othello_gpt.pth")