from transformer_lens import HookedTransformer, HookedTransformerConfig
from transformer_lens.train import train, HookedTransformerTrainConfig
from datasets import Dataset
import numpy as np
import torch as t

def train_othello_gpt(d_model, n_layers, lr, batch_size, num_epochs, wandb):
    device = t.device("cuda" if t.cuda.is_available() else "cpu")

    model_cfg = HookedTransformerConfig(
        n_layers=n_layers,
        d_model=d_model,
        d_head=64,
        n_heads=8,
        d_mlp=d_model * 4,
        d_vocab=61,
        n_ctx=59,
        act_fn="gelu",
        normalization_type="LN",
        device=device,
    )

    model = HookedTransformer(model_cfg).to(device)

    train_cfg = HookedTransformerTrainConfig(
        lr=lr,
        batch_size=batch_size,
        num_epochs=num_epochs,
        device=device,
        wandb=wandb,
        wandb_project_name="OthelloGPTTraining",
    )

    tokenized_data = t.tensor(np.load("data/board_seqs_int_small.npy"), dtype=t.long)
    tokenized_data = tokenized_data[:, :59]  # remove XX at the end
    data_dict = {"tokens": tokenized_data.tolist()}

    dataset = Dataset.from_dict(data_dict)
    dataset.set_format(type="torch", columns=["tokens"])

    train(model, train_cfg, dataset)
    t.save(model.state_dict(), f"othello_gpt_{n_layers}_{d_model}_lr{lr}_bs{batch_size}_epochs{num_epochs}.pt")

