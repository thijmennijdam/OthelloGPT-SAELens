import argparse
from train_othello_gpt import train_othello_gpt

def main():
    parser = argparse.ArgumentParser(description="Train Othello GPT Model")
    parser.add_argument('--d_model', type=int, required=True, help='Dimension of the model')
    parser.add_argument('--n_layers', type=int, required=True, help='Number of layers in the model')
    parser.add_argument('--lr', type=float, required=True, help='Learning rate')
    parser.add_argument('--batch_size', type=int, required=True, help='Batch size')
    parser.add_argument('--epochs', type=int, required=True, help='Number of epochs')
    parser.add_argument('--wandb', action='store_true', help='Use Weights and Biases for logging')

    args = parser.parse_args()

    train_othello_gpt(
        d_model=args.d_model,
        n_layers=args.n_layers,
        lr=args.lr,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        wandb=args.wandb,
    )

if __name__ == "__main__":
    main()