import argparse
from sae_lens import LanguageModelSAERunnerConfig, SAETrainingRunner
import torch as t

def main(args):
    total_training_steps = args.total_training_steps
    batch_size = args.batch_size
    total_training_tokens = total_training_steps * batch_size

    lr_warm_up_steps = 0
    lr_decay_steps = total_training_steps // 5  # 20% of training
    l1_warm_up_steps = total_training_steps // 20  # 5% of training

    layer = args.layer

    cfg = LanguageModelSAERunnerConfig(
        # Data Generating Function (Model + Training Distibuion)
        model_name="my-own-othello-model",  # added this to official list in TransformerLens library
        hook_name=f"blocks.{layer}.hook_resid_pre",  # A valid hook point
        hook_layer=layer,  # Only one layer in the model.
        d_in=128,  # the width the hook point
        dataset_path="taufeeque/othellogpt",  # my own dataset which I created in this file and uploaded to HF.
        is_dataset_tokenized=True, # dataset is tokenized, although I saw this flag is not in use anymore
        streaming=False,  # we could pre-download the token dataset if it was small.
        prepend_bos=False,
        # SAE Parameters
        mse_loss_normalization=None,  # We won't normalize the mse loss,
        expansion_factor=args.expansion_factor,  # the width of the SAE. Larger will result in better stats but slower training.
        b_dec_init_method="zeros",  # The geometric median can be used to initialize the decoder weights.
        apply_b_dec_to_input=False,  # We won't apply the decoder weights to the input.
        normalize_sae_decoder=False,
        scale_sparsity_penalty_by_decoder_norm=True,
        decoder_heuristic_init=True,
        init_encoder_as_decoder_transpose=True,
        # normalize_activations=True,
        # Training Parameters
        lr=args.lr,  # lower the better, we'll go fairly high to speed up the tutorial.
        adam_beta1=0.9,  # adam params (default, but once upon a time we experimented with these.)
        adam_beta2=0.999,
        lr_scheduler_name="constant",  # constant learning rate with warmup. Could be better schedules out there.
        lr_warm_up_steps=lr_warm_up_steps,  # this can help avoid too many dead features initially.
        lr_decay_steps=lr_decay_steps,  # this will help us avoid overfitting.
        l1_coefficient=args.l1_coefficient,  # will control how sparse the feature activations are
        l1_warm_up_steps=l1_warm_up_steps,  # this can help avoid too many dead features initially.
        lp_norm=1.0,  # the L1 penalty (and not a Lp for p < 1)
        train_batch_size_tokens=batch_size,
        context_size=59,  # will control the length of the prompts we feed to the model. Larger is better but slower. so for the tutorial we'll use a short one.
        # Activation Store Parameters
        n_batches_in_buffer=64,  # controls how many activations we store / shuffle.
        training_tokens=total_training_tokens,  # 100 million tokens is quite a few, but we want to see good stats. Get a coffee, come back.
        store_batch_size_prompts=16,
        # Resampling protocol
        use_ghost_grads=False,  # we don't use ghost grads anymore.
        feature_sampling_window=1000,  # this controls our reporting of feature sparsity stats
        dead_feature_window=1000,  # would affect resampling or ghost grads if we were using it.
        dead_feature_threshold=1e-4,  # would affect resampling or ghost grads if we were using it.
        # WANDB
        log_to_wandb=True,  # always use wandb unless you are just testing code.
        wandb_project="sae-othello",  # the project name in wandb.
        wandb_log_frequency=30,
        eval_every_n_wandb_logs=20,
        # Misc
        # device=device,
        seed=42,
        n_checkpoints=20,
        checkpoint_path="checkpoints",
        dtype="float32"
    )
    # look at the next cell to see some instruction for what to do while this is running.
    sparse_autoencoder = SAETrainingRunner(cfg).run()

    path = f"new/new_trained_sae_{cfg.model_name}_{cfg.hook_name}_{cfg.hook_layer}_{cfg.expansion_factor}_{cfg.l1_coefficient}_{cfg.train_batch_size_tokens}_{cfg.training_tokens}_{cfg.lr}_{cfg.l1_coefficient}"

    sparse_autoencoder.save_model(path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train SAE model with specified parameters.')
    parser.add_argument('--total_training_steps', type=int, required=True, help='Total training steps')
    parser.add_argument('--batch_size', type=int, required=True, help='Batch size')
    parser.add_argument('--layer', type=int, required=True, help='Layer')
    parser.add_argument('--expansion_factor', type=int, required=True, help='Expansion factor')
    parser.add_argument('--l1_coefficient', type=float, required=True, help='L1 coefficient')
    parser.add_argument('--lr', type=float, required=True, help='Learning rate')

    args = parser.parse_args()
    main(args)
