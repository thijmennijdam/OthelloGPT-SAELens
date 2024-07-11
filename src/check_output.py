from transformer_lens import HookedTransformer, HookedTransformerConfig
import torch as t
from datasets import load_from_disk
import numpy as np
from utils.plotly_utils import imshow
from neel_plotly import scatter, line

OTHELLO_MECHINT_ROOT = './othello_world/mechanistic_interpretability'

from othello_world.mechanistic_interpretability.mech_interp_othello_utils import (
    plot_board,
    plot_single_board,
    plot_board_log_probs,
    to_string,
    to_int,
    int_to_label,
    string_to_label,
    OthelloBoardState
)



# model_name = "othello_gpt_6_128_lr0.005_bs512_epochs4_LNPre"
# model = t.load(f"./{model_name}.pt")
# t.save(model, f"./models/othello/{model_name}.pth")




# model_name = "othello_gpt_6_128_lr0.005_bs512_epochs4_LNPre"
# model = t.load(f"./{model_name}.pt")
# t.save(model, f"./models/othello/{model_name}.pth")


n_layers = 6
d_model = 128

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

# save model
t.save(model, "randomly_initialized_othello_gpt_6_128.pth")
# save model
t.save(model, "randomly_initialized_othello_gpt_6_128.pth")

model.load_and_process_state_dict(t.load(f"./models/othello/{model_name}.pth"), fold_ln=False)
model.load_and_process_state_dict(t.load(f"./models/othello/{model_name}.pth"), fold_ln=False)

# # DIFFERENT MODEL
# cfg = HookedTransformerConfig(
#     n_layers = 8,
#     d_model = 512,
#     d_head = 64,
#     n_heads = 8,
#     d_mlp = 2048,
#     d_vocab = 61,
#     n_ctx = 59,
#     act_fn="gelu",
#     normalization_type="LNPre",
#     device=device,
# )
# model = HookedTransformer(cfg)

# import transformer_lens.utils as utils

# sd = utils.download_file_from_hf("NeelNanda/Othello-GPT-Transformer-Lens", "synthetic_model.pth")
# # champion_ship_sd = utils.download_file_from_hf("NeelNanda/Othello-GPT-Transformer-Lens", "championship_model.pth")
# model.load_state_dict(sd)
        
        
# # load as pt, and save as pth
# model = t.load(f"./models/othello_gpt_6_128_lr0.0005_bs512_epochs2.pth")

# # save as pth
# t.save(model, f"./models/othello_gpt_6_128_lr0.0005_bs512_epochs2.pth")


# cfg_dict = model_cfg.to_dict()
# print(cfg_dict)
# cfg_dict = model_cfg.to_dict()
# print(cfg_dict)
# save cfg_dict as json
# import json
# with open("models/othello_gpt_6_128_lr0.0005_bs512_epochs2.json", "w") as f:
#     json.dump(cfg_dict, f)


dataset = load_from_disk("data/validation_processed_othellogpt_dataset")

# Load board data as ints (i.e. 0 to 60)
board_seqs_int = t.tensor(np.load(f"{OTHELLO_MECHINT_ROOT}/board_seqs_int_small.npy"), dtype=t.long)
# Load board data as "strings" (i.e. 0 to 63 with middle squares skipped out)
board_seqs_string = t.tensor(np.load(f"{OTHELLO_MECHINT_ROOT}/board_seqs_string_small.npy"), dtype=t.long)

assert all([middle_sq not in board_seqs_string for middle_sq in [27, 28, 35, 36]])
assert board_seqs_int.max() == 60

num_games, length_of_game = board_seqs_int.shape
print("Number of games:", num_games)
print("Length of game:", length_of_game)

# Define possible indices (excluding the four center squares)
stoi_indices = [i for i in range(64) if i not in [27, 28, 35, 36]]

# Define our rows, and the function that converts an index into a (row, column) label, e.g. `E2`
alpha = "ABCDEFGH"

def to_board_label(i):
    return f"{alpha[i//8]}{i%8}"

# Get our list of board labels

board_labels = list(map(to_board_label, stoi_indices))
full_board_labels = list(map(to_board_label, range(64)))

moves_int = board_seqs_int[0, :30]

# This is implicitly converted to a batch of size 1
logits = model(moves_int)
print("logits:", logits.shape)

logit_vec = logits[0, -1]
log_probs = logit_vec.log_softmax(-1)
# Remove the "pass" move (the zeroth vocab item)
log_probs = log_probs[1:]
assert len(log_probs)==60

# Set all cells to -13 by default, for a very negative log prob - this means the middle cells don't show up as mattering
temp_board_state = t.zeros((8, 8), dtype=t.float32, device=device) - 13.
temp_board_state.flatten()[stoi_indices] = log_probs


# def plot_square_as_board(state, diverging_scale=True, **kwargs):
#     '''Takes a square input (8 by 8) and plot it as a board. Can do a stack of boards via facet_col=0'''
#     kwargs = {
#         "y": [i for i in alpha],
#         "x": [str(i) for i in range(8)],
#         "color_continuous_scale": "RdBu" if diverging_scale else "Blues",
#         "color_continuous_midpoint": 0. if diverging_scale else None,
#         "aspect": "equal",
#         **kwargs
#     }
#     imshow(state, **kwargs)
    


# plot_square_as_board(temp_board_state.reshape(8, 8), zmax=0, diverging_scale=False, title="Example Log Probs")


# fig = plot_single_board(int_to_label(moves_int), return_fig=True)
# fig.write_image("othello_board_example.png")


num_games = 100
focus_games_int = board_seqs_int[:num_games]
focus_games_string = board_seqs_string[:num_games]

print("focus_games_int")
print(focus_games_int)


print("focus_games_string")
print(focus_games_string)
print(focus_games_string.shape)
def one_hot(list_of_ints, num_classes=64):
    out = t.zeros((num_classes,), dtype=t.float32)
    out[list_of_ints] = 1.
    return out


focus_states = np.zeros((num_games, 60, 8, 8), dtype=np.float32)
focus_valid_moves = t.zeros((num_games, 60, 64), dtype=t.float32)

for i in (range(num_games)):
    board = OthelloBoardState()
    for j in range(60):
        board.umpire(focus_games_string[i, j].item())
        focus_states[i, j] = board.state
        focus_valid_moves[i, j] = one_hot(board.get_valid_moves())

print("focus states:", focus_states.shape)
print("focus_valid_moves", tuple(focus_valid_moves.shape))


focus_logits, focus_cache = model.run_with_cache(focus_games_int[:, :-1].to(device))
focus_preds = focus_logits.argmax(-1) # shape: (num_games, 59)

# convert preds to label
focus_preds = t.tensor(to_string(focus_preds))
print(focus_preds.shape)
# convert preds to label
focus_preds = t.tensor(to_string(focus_preds))
print(focus_preds.shape)
# for i in range(10):
#     print("valid moves for game 1, move", i)
#     print(t.where(focus_valid_moves[0, i, :] == 1))
#     print("predicted move", focus_preds[0, i])
#     print('move is correct:', focus_valid_moves[0, i, focus_preds[0, i] - 1] == 1)

correct_predictions = 0
total_predictions = 0

for game in range(100):  # Iterate over the first 100 games
    for move in range(focus_valid_moves.shape[1] - 1):  # Iterate over all moves
        if focus_valid_moves[game, move, focus_preds[game, move]] == 1:
        if focus_valid_moves[game, move, focus_preds[game, move]] == 1:
            correct_predictions += 1
        total_predictions += 1

accuracy = correct_predictions / total_predictions
print("Accuracy over the first 100 games and all moves:", accuracy)

# print predictions for game 1, move 1
# print(focus_preds[0, :5])    




# # get argmax to see which move is predicted
# print(focus_logits.shape)


# print("focus_preds")
# print(focus_preds.shape)
# print(focus_preds)

imshow(
    focus_states[0, :16],
    filename='plots/game1',
    renderer=None,
    facet_col=0,
    facet_col_wrap=8,
    facet_labels=[f"Move {i}" for i in range(1, 17)],
    title="First 16 moves of first game",
    color_continuous_scale="Greys",
)


imshow(
    focus_states[0, :16],
    filename='plots/game2',
    renderer=None,
    facet_col=0,
    facet_col_wrap=8,
    facet_labels=[f"Move {i}" for i in range(1, 17)],
    title="First 16 moves of first game",
    color_continuous_scale="Greys",
)





# convert to string
# focus_preds_string = int_to_label(focus_preds)
# print("focus_preds_string")
# print(focus_preds_string)

# focus_logits.shape

# layer = 4
# neuron = 100

# neuron_acts = focus_cache["post", layer, "mlp"][:, :, neuron]

# imshow(
#     neuron_acts,
#     title=f"L{layer}N{neuron} Activations over 50 games",
#     labels={"x": "Move", "y": "Game"},
#     aspect="auto",
#     width=900
# )

# top_moves = neuron_acts > neuron_acts.quantile(0.99)

def state_stack_to_one_hot(state_stack):
    '''
    Creates a tensor of shape (games, moves, rows=8, cols=8, options=3), where the [g, m, r, c, :]-th entry
    is a one-hot encoded vector for the state of game g at move m, at row r and column c. In other words, this
    vector equals (1, 0, 0) when the state is empty, (0, 1, 0) when the state is "their", and (0, 0, 1) when the
    state is "my".
    '''
    one_hot = t.zeros(
        state_stack.shape[0], # num games
        state_stack.shape[1], # num moves
        8,
        8,
        3, # the options: empty, white, or black
        device=state_stack.device,
        dtype=t.int,
    )
    one_hot[..., 0] = state_stack == 0 
    one_hot[..., 1] = state_stack == -1 
    one_hot[..., 2] = state_stack == 1 

    return one_hot

# We first convert the board states to be in terms of my (+1) and their (-1), rather than black and white

# print("focus_games_int")
# print(focus_games_int)
# alternating = np.array([-1 if i%2 == 0 else 1 for i in range(focus_games_int.shape[1])])

# print("alternating")
# print(alternating)

# flipped_focus_states = focus_states * alternating[None, :, None, None]

# print("flipped_focus_states")
# print(flipped_focus_states)


# # We now convert to one-hot encoded vectors
# focus_states_flipped_one_hot = state_stack_to_one_hot(t.tensor(flipped_focus_states))

# print("focus_states_flipped_one_hot")
# print(focus_states_flipped_one_hot.shape)
# print(focus_states_flipped_one_hot)

# # Take the argmax (i.e. the index of option empty/their/mine)
# focus_states_flipped_value = focus_states_flipped_one_hot.argmax(dim=-1)

# print("focus_states_flipped_value")
# print(focus_states_flipped_value.shape)
# print(focus_states_flipped_value)

# focus_states_flipped_value = focus_states_flipped_value.to(device)

# board_state_at_top_moves = t.stack([
#     (focus_states_flipped_value == 2)[:, :-1][top_moves].float().mean(0),
#     (focus_states_flipped_value == 1)[:, :-1][top_moves].float().mean(0),
#     (focus_states_flipped_value == 0)[:, :-1][top_moves].float().mean(0)
# ])

# print("board_state_at_top_moves")
# print(board_state_at_top_moves.shape)
# print(board_state_at_top_moves)

# plot_square_as_board(
#     board_state_at_top_moves, 
#     facet_col=0,
#     facet_labels=["Mine", "Theirs", "Blank"],
#     title=f"Aggregated top 30 moves for neuron L{layer}N{neuron}", 
# )