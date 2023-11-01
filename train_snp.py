# %%
import os
import sys
import numpy as np
import torch
import math
import matplotlib.pyplot as plt
import seaborn as sns
from torch import Tensor
from torch.nn.modules.module import Module
import cProfile
from torch.utils.data import Dataset, DataLoader
import transformer_lens as tl
import json
import time
import pstats
from einops import repeat
from torch.profiler import profile, record_function, ProfilerActivity
import wandb
import argparse
import gc

from snp_utils import HookedKataGoWrapper, mask_flippedness
sys.path.append("./KataGo/python")

from sgfmill import sgf
from KataGo.python.board import Board
from KataGo.python.data import Metadata, load_sgf_moves_exn
# from KataGo.python.play import get_outputs
from KataGo.python.load_model import load_model
from KataGo.python.features import Features
from tqdm import tqdm


# %%

DATASET_DIR = 'dataset'
CHECKPOINT_FILE = 'kg_checkpoint/kata1-b18c384nbt-s7709731328-d3715293823/model.ckpt'
DEVICE = 'cuda'
pos_len = 19

parser = argparse.ArgumentParser(
    description="Use modified subnetwork probing to train the KataGo policy net to make the worst move")
add = parser.add_argument

add("--dataset_dir", default=DATASET_DIR)
add("--checkpoint_file", default=CHECKPOINT_FILE)
add("--device", default=DEVICE)
add("--wandb", default=True, type=bool)
add("--n_epochs", default=100, type=int)
add("--lr", default=0.01, type=float)
add("--regularization_lambda", default=1, type=float)

args, unknown = parser.parse_known_args()
print(f"args: {args}")

# %%

kata_model, swa_model, _ = load_model(args.checkpoint_file, None, device=args.device, pos_len=pos_len, verbose=True)
kata_model.eval()
model_config = kata_model.config
if swa_model is not None:
    kata_model = swa_model.module
    kata_model.eval()

# %%

class KataPessimizeDataset(Dataset):
    def __init__(self, data_dir, n_games:None, moves_per_game:int=100):
        self.data_dir = data_dir
        self.moves_per_game = moves_per_game
        self.games = []
        for filename in os.listdir(data_dir)[:n_games]:
            if filename.endswith(".npz"):
                with np.load(os.path.join(data_dir, filename)) as data:
                    self.games.append({
                        "bin_input_data": data['bin_input_data'],
                        "global_input_data": data['global_input_data'],
                        "annotated_values": data['annotated_values'],
                        "pla": data['pla'],
                    })
        print(f"Loaded {len(self.games)} games from {data_dir}")

    def __len__(self):
        return len(self.games) * self.moves_per_game
    
    def __getitem__(self, idx):
        game = self.games[idx // self.moves_per_game]
        move = idx % self.moves_per_game
        return {
            "bin_input_data": game['bin_input_data'][move],
            "global_input_data": game['global_input_data'][move],
            "annotated_values": game['annotated_values'][move],
            "pla": game['pla'][move],
        }
    

dataset = KataPessimizeDataset(args.dataset_dir, n_games=800)
val_frac = 0.1
train_set, val_set = torch.utils.data.random_split(dataset, [int(len(dataset)*(1-val_frac)), int(len(dataset)*val_frac)])

# %%
wrapped_model = HookedKataGoWrapper(kata_model).to(args.device)
wrapped_model.freeze_weights()

# %%

# %%
features = Features(model_config, pos_len)
board = Board(19)

loc_to_move_map = torch.zeros(362, dtype=torch.int64)
for i in range(board.arrsize):
    entry = features.loc_to_tensor_pos(i, board)
    if 0 <= entry < 362:
        loc_to_move_map[entry] = i

# %%

# copied from play
def get_policy(model:HookedKataGoWrapper, bin_input_data, global_input_data, annotated_values=None):
    """
    Runs the KataGo model on the given input data and returns the policy and value.
    """
    # print(f"bin_input_data.shape {bin_input_data.shape}")
    # print(f"global_input_data.shape {global_input_data.shape}")
    model.eval()
    batch_size = bin_input_data.shape[0]

    # Currently we don't actually do any symmetries
    # symmetry = 0
    # model_outputs = model(apply_symmetry(batch["binaryInputNCHW"],symmetry),batch["globalInputNC"])

    model_outputs = model(
        bin_input_data.to(DEVICE),
        global_input_data.to(DEVICE),
    )

    outputs = model.mod.postprocess_output(model_outputs)
    (
        policy_logits,      # N, num_policy_outputs, move
        value_logits,       # N, {win,loss,noresult}
        td_value_logits,    # N, {long, mid, short} {win,loss,noresult}
        pred_td_score,      # N, {long, mid, short}
        ownership_pretanh,  # N, 1, y, x
        pred_scoring,       # N, 1, y, x
        futurepos_pretanh,  # N, 2, y, x
        seki_logits,        # N, 4, y, x
        pred_scoremean,     # N
        pred_scorestdev,    # N
        pred_lead,          # N
        pred_variance_time, # N
        pred_shortterm_value_error, # N
        pred_shortterm_score_error, # N
        scorebelief_logits, # N, 2 * (self.pos_len*self.pos_len + EXTRA_SCORE_DISTR_RADIUS)
    ) = (x for x in outputs[0]) # batch

    nan_mask = torch.isnan(annotated_values)
    reshaped_nan_mask = torch.gather(nan_mask, 1, repeat(loc_to_move_map, 'loc -> batch loc', batch=batch_size))
    policy0_logits = policy_logits[:, 0, :] # batch, loc
    policy0_logits[reshaped_nan_mask] = -torch.inf # Prevent policy from making illegal moves

    policy0 = torch.nn.functional.softmax(policy0_logits,dim=1) # batch, loc
    # policy_inverted = torch.nn.functional.softmax(- policy_logits[0,:], dim=0)
    value = torch.nn.functional.softmax(value_logits,dim=0).detach().cpu().numpy()

    # board = gs.board

    # Assume that all moves are legal and handle illegal moves in loss_fn
    probs0 = torch.zeros((batch_size, board.arrsize))
    for batch_idx in range(batch_size):
        for i in range(policy0.shape[1]):
            move = features.tensor_pos_to_loc(i,board)
            if i == policy0.shape[1]-1: # pass move
                # pass
                probs0[batch_idx,Board.PASS_LOC] = policy0[batch_idx,i]
            else: # elif board.would_be_legal(board.pla,move):
                probs0[batch_idx,move] = policy0[batch_idx,i]

    # probs_inverted = torch.zeros(board.arrsize)
    # for i in range(len(policy_inverted)):
    #     move = features.tensor_pos_to_loc(i,board)
    #     if i == len(policy_inverted)-1:
    #         pass
    #         # probs_inverted[Board.PASS_LOC] = policy_inverted[i]
    #     else: # elif board.would_be_legal(board.pla,move):
    #         probs_inverted[move] = policy_inverted[i]

    nan_probs = torch.isnan(probs0).sum()
    if nan_probs > 0:
        print(f"Warning: probs0 has {nan_probs} nan values")

    return {
        "policy0": policy0,
        # "policy_inverted": policy_inverted,
        # "policy1": policy1,
        "probs0": probs0,
        # "probs_inverted": probs_inverted,
        # "moves_and_probs1": moves_and_probs1,
        "value": value,

        # "td_value": td_value,
        # "td_value2": td_value2,
        # "td_value3": td_value3,
        # "scoremean": scoremean,
        # "td_score": td_score,
        # "scorestdev": scorestdev,
        # "lead": lead,
        # "vtime": vtime,
        # "estv": estv,
        # "ests": ests,
        # "ownership": ownership,
        # "ownership_by_loc": ownership_by_loc,
        # "scoring": scoring,
        # "scoring_by_loc": scoring_by_loc,
        # "futurepos": futurepos,
        # "futurepos0_by_loc": futurepos0_by_loc,
        # "futurepos1_by_loc": futurepos1_by_loc,
        # "seki": seki,
        # "seki_by_loc": seki_by_loc,
        # "seki2": seki2,
        # "seki_by_loc2": seki_by_loc2,
        # "scorebelief": scorebelief,
        # "genmove_result": genmove_result
    }
# %%

def print_board_template(arr, size=19):
    board = Board(size)
    board.board = arr
    # result = np.zeros((size+2, size+1), dtype=int)
    # for y in range(size + 2):
    #     for x in range(size + 1):
    #         board[y, x] = arr[board.loc(x-1, y-1)]
    print(board.to_string())
    print()

# print_board_template(19)

# %%
def loss_fn(policy_probs:Tensor, annotated_values:Tensor, pla:int):
    """
    Calculates regret for a given policy.
    Here regret means how far we are off the WORST move.
    Inputs:
        policy_probs: Tensor of shape (batch_size, board.arrsize)
        annotated_values: Tensor of shape (batch_size, board.arrsize)
        pla: array of (hopefully) 1s and 2s, shape (batch_size,)
    """
    if policy_probs.shape != annotated_values.shape:
        raise Exception(f"policy_probs.shape {policy_probs.shape} != annotated_values.shape {annotated_values.shape}")
    bad_pla_mask = ~((pla == 1) | (pla == 2))
    if bad_pla_mask.any():
        raise ValueError(f"pla has values other than 1 or 2 at {bad_pla_mask.nonzero().tolist()}")
        annotated_values[bad_pla_mask] = 0
    
    annotated_values[pla == 2] = 1 - annotated_values[pla == 2]
    # Policy should be 0 wherever annotated value is nan
    nan_mask = torch.isnan(annotated_values)
    # print(f"nan values: {nan_mask.sum(dim=-1)}")
    # for i in range(nan_mask.shape[0]):
    #     total_nonnan_prob = policy_probs[i][~nan_mask[i]].sum()
    #     if total_nonnan_prob < 0.5:
    #         print(f"Warning: total_nonnan_prob={total_nonnan_prob} at {i}")
    #         print(f"nans")
    #         print_board_template(nan_mask[i])
    #         # print(f"{policy_probs[i]=}")
    #         policy_mask = (policy_probs[i] > 1e-3)
    #         print(f"policy > 1e-3:")
    #         print_board_template(policy_mask)
    #         print(f"policy & nans:")
    #         print_board_template(policy_mask & nan_mask[i])
        # if policy_probs[i][nan_mask].nonzero().shape[0] != 0:
        #     bad_list = policy_probs[i][nan_mask].nonzero()
        #     print(f"Warning: policy_probs has non-zero values where annotated_values is nan, at {bad_list.numel()} locations {bad_list.tolist()}")

    annotated_values = torch.nan_to_num(annotated_values, nan=torch.inf)
    annotated_values = annotated_values - annotated_values.min() # regrets
    annotated_values[nan_mask] = 0
    policy_probs[nan_mask] = 0 # Assume policy never makes illegal moves
    if (policy_probs.sum(dim=-1) - 1).abs().max() > 1e-3:
        print(f"Warning: sum(probs) != 1; range={policy_probs.sum(dim=-1).min(), policy_probs.sum(dim=-1).max()}")
    policy_probs *= 1 / policy_probs.sum(dim=-1, keepdim=True)
    result = (policy_probs * annotated_values).sum(dim=-1)
    return result

def get_losses(model, batch):
    bin_input_data, global_input_data, pla = batch["bin_input_data"], batch["global_input_data"], batch["pla"]
    policy_data = get_policy(model, bin_input_data, global_input_data, batch["annotated_values"])
    losses = loss_fn(policy_data["probs0"], batch["annotated_values"], pla)
    return losses


# %%

def train(wrapped_model:HookedKataGoWrapper, train_loader:DataLoader, val_loader: DataLoader, n_epochs=1, regularization_lambda=1, lr=0.005, use_wandb=True):
    print(f"Starting training for {n_epochs} epochs")
    print(f"with regularization_lambda={regularization_lambda}, lr={lr}")
    if use_wandb:
        wandb.init(project="kata-pessimize")
        wandb.watch(wrapped_model)
    optimizer = torch.optim.Adam(wrapped_model.parameters(), lr=lr) # TODO change lr
    with wrapped_model.with_fwd_hooks() as hooked_model:
        for epoch in range(n_epochs):
            regrets = []
            regularization_losses = []
            total_losses = []
            hooked_model.train()
            print(f"Starting epoch {epoch}/{n_epochs}")
            for batch in tqdm(train_loader):
                optimizer.zero_grad()
                losses = get_losses(hooked_model, batch)
                regularization_loss = hooked_model.regularization_loss()
                avg_loss = losses.mean()
                if epoch==0: print(avg_loss.item())
                total_loss = regularization_lambda * regularization_loss + avg_loss
                total_loss.backward()
                optimizer.step()
                regrets.append(avg_loss.item())
                regularization_losses.append(regularization_loss.item())
                total_losses.append(total_loss.item())
            print(f"Average regret: {np.mean(regrets)}")
            print(f"Average regularization loss: {np.mean(regularization_losses)}")
            mean_mask_value = torch.tensor([wrapped_model.sample_mask(n).mean() for n in wrapped_model.mask_logits_names]).mean()
            if use_wandb:
                val_regrets = []
                hooked_model.eval()
                for batch in tqdm(val_loader):
                    losses = get_losses(hooked_model, batch)
                    val_regrets.append(losses.mean().item())
                print(f"Average validation regret: {np.mean(val_regrets)}")
                wandb.log({"regret": np.mean(regrets), "val_regret": np.mean(val_regrets), "regularization_loss": np.mean(regularization_losses),
                    "total_loss": np.mean(total_losses), "mean_mask_value": mean_mask_value, "flippedness": mask_flippedness(wrapped_model).sum()})
            if (epoch + 1) % 5 == 0:
                time_str = time.strftime("%Y%m%d-%H%M%S")
                state_dict = wrapped_model.state_dict()
                state_dict["config"] = model_config
                torch.save(state_dict, f"models/model_{epoch}_{time_str}.pth")

    if use_wandb:
        wandb.finish()


# %%

# %%
train_loader = DataLoader(train_set, batch_size=96, shuffle=True, num_workers=0)
val_loader = DataLoader(val_set, batch_size=96, shuffle=True, num_workers=0)
# %%
gc.collect()
torch.cuda.empty_cache()
train(wrapped_model, train_loader, val_loader, n_epochs=int(args.n_epochs), lr=args.lr, regularization_lambda=args.regularization_lambda, use_wandb=args.wandb)
# cProfile.run("train(wrapped_model, data_loader)", "output.pstats")
