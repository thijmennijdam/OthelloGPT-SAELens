from transformer_lens import HookedTransformer, HookedTransformerConfig
from transformer_lens.train import train, HookedTransformerTrainConfig
from datasets import Dataset, load_dataset, load_from_disk
import numpy as np
import torch as t
from tqdm import tqdm


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
        normalization_type="LNPre",
        device=device,
    )

    model = HookedTransformer(model_cfg).to(device)

    train_cfg = HookedTransformerTrainConfig(
        lr=lr,
        batch_size=batch_size,
        num_epochs=num_epochs,
        optimizer_name="AdamW",
        weight_decay=0.01,
        device=device,
        wandb=wandb,
        wandb_project_name="OthelloGPTTraining",
    )

    # tokenized_data = t.tensor(np.load("data/board_seqs_int_small.npy"), dtype=t.long)
    # tokenized_data = tokenized_data[:, :59]  # remove XX at the end
    # data_dict = {"tokens": tokenized_data.tolist()}
    # dataset = Dataset.from_dict(data_dict)
    # dataset.set_format(type="torch", columns=["tokens"])
    
    # streamed_dataset = load_dataset("taufeeque/othellogpt", split="train", streaming=True)


    # Load the dataset in streaming mode
    # streamed_dataset = load_dataset("taufeeque/othellogpt", split="validation", streaming=True)
    dataset = load_from_disk("data/processed_othellogpt_dataset")
    # Function to process each example
    # def truncate_tokens(example):
    #     example['tokens'] = example['tokens'][:-1]
    #     return example

    # # Collect the first 100,000 samples and apply the truncation
    # processed_tokens = []
    # max_samples = 400_000

    # for i, example in enumerate(tqdm(streamed_dataset, total=max_samples, desc="Processing dataset")):
    #     if i >= max_samples:
    #         break
    #     processed_example = truncate_tokens(example)
    #     processed_tokens.append(processed_example['tokens'])

    # # Convert the list of processed tokens into a Dataset
    # data_dict = {"tokens": processed_tokens}
    # processed_dataset = Dataset.from_dict(data_dict)

    # # Set format to PyTorch
    # processed_dataset.set_format(type="torch", columns=["tokens"])

    # # Save the processed dataset locally
    # processed_dataset.save_to_disk(f"{max_samples}_training_processed_othellogpt_dataset")
    
    train(model, train_cfg, dataset)
    t.save(model.state_dict(), f"othello_gpt_{n_layers}_{d_model}_lr{lr}_bs{batch_size}_epochs{num_epochs}_LNPre.pt")

