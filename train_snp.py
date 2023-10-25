# %%
"""
Converts the SGF files into a training set that can be passed to the model,
which has a position and the value for each possible move.
Then applies subnetwork probing (TODO)
"""

import os
import sys
import numpy as np
import torch
import math
import matplotlib.pyplot as plt
import seaborn as sns
from torch import Tensor
from torch.nn.modules.module import Module
from torch.utils.data import Dataset, DataLoader
import transformer_lens as tl
import json

from snp_utils import HookedKataGoWrapper
sys.path.append("/home/ubuntu/katago_pessimize/KataGo/python")

from sgfmill import sgf
from KataGo.python.board import Board
from KataGo.python.data import Metadata, load_sgf_moves_exn
# from KataGo.python.play import get_outputs
from KataGo.python.load_model import load_model
from KataGo.python.features import Features
from tqdm import tqdm


# %%

SGF_DIR = 'sgf_downloads'
DATASET_DIR = 'dataset'
ANNOTATIONS_DIR = 'annotations'
CHECKPOINT_FILE = 'kg_checkpoint/kata1-b18c384nbt-s7709731328-d3715293823/model.ckpt'
DEVICE = 'cuda'
pos_len = 19

kata_model, swa_model, _ = load_model(CHECKPOINT_FILE, None, device=DEVICE, pos_len=pos_len, verbose=True)
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
                    print(data)
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
    

dataset = KataPessimizeDataset(DATASET_DIR, n_games=60)

data_loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=0)

# %%
wrapped_model = HookedKataGoWrapper(kata_model)

# %%

# %%
features = Features(model_config, pos_len)
board = Board(19)

# copied from play
def get_policy(model:HookedKataGoWrapper, bin_input_data, global_input_data):
    print(f"bin_input_data.shape {bin_input_data.shape}")
    print(f"global_input_data.shape {global_input_data.shape}")
    with torch.no_grad():
        model.eval()
        batch_size = bin_input_data.shape[0]

        # Currently we don't actually do any symmetries
        # symmetry = 0
        # model_outputs = model(apply_symmetry(batch["binaryInputNCHW"],symmetry),batch["globalInputNC"])


        with model.hooks() as hooked_model:
            model_outputs = hooked_model.mod(
                torch.tensor(bin_input_data, dtype=torch.float32).to(DEVICE),
                torch.tensor(global_input_data, dtype=torch.float32).to(DEVICE),
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

        print(f"{policy_logits.shape=}")
        policy0 = torch.nn.functional.softmax(policy_logits[:, 0, :],dim=1) # batch, move
        # policy_inverted = torch.nn.functional.softmax(- policy_logits[0,:], dim=0)
        value = torch.nn.functional.softmax(value_logits,dim=0).cpu().numpy()


        # board = gs.board

        # Assume that all moves are legal and handle illegal moves in loss_fn
        probs0 = torch.zeros((batch_size, board.arrsize))
        print(f"policy0.shape {policy0.shape}")
        for batch_idx in range(batch_size):
            for i in range(len(policy0[batch_idx])):
                move = features.tensor_pos_to_loc(i,board)
                if i == len(policy0)-1:
                    pass
                    # probs0[Board.PASS_LOC] = policy0[i]
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
def loss_fn(policy_probs:Tensor, annotated_values:Tensor, pla:int):
    """
    Calculates regret for a given policy.
    Here regret means how far we are off the WORST move.
    """
    if policy_probs.shape != annotated_values.shape:
        raise Exception(f"policy_probs.shape {policy_probs.shape} != annotated_values.shape {annotated_values.shape}")
    bad_pla_mask = ~((pla == 1) | (pla == 2))
    if bad_pla_mask.any():
        print(f"Warning: pla has values other than 1 or 2 at {bad_pla_mask.nonzero().tolist()}")
        # TODO do something about this
    
    if pla == 2: # all annotated values are black's winrate
        annotated_values = 1 - annotated_values
    # Policy should be 0 wherever annotated value is nan
    nan_mask = torch.isnan(annotated_values)
    # if policy_probs[nan_mask].nonzero().shape[0] != 0:
    #     bad_list = policy_probs[nan_mask].nonzero()
    #     print(f"Warning: policy_probs has non-zero values where annotated_values is nan, at {bad_list.numel()} locations {bad_list.tolist()}")

    annotated_values = torch.nan_to_num(annotated_values, nan=torch.inf)
    annotated_values = annotated_values - annotated_values.min() # regrets
    annotated_values[nan_mask] = 0
    policy_probs[nan_mask] = 0 # Assume policy never makes illegal moves
    if (policy_probs.sum() - 1).abs() > 1e-4:
        print(f"Warning: sum(probs) != 1; {policy_probs.sum()}")
    policy_probs *= 1 / policy_probs.sum()
    return (policy_probs * annotated_values).sum()

# %%

def print_board_template(size=19):
    board = Board(size)
    result = np.zeros((size+2, size+1), dtype=int)
    for y in range(size + 2):
        for x in range(size + 1):
            result[y, x] = board.loc(x-1, y-1)
    print(result)

# print_board_template(19)


# %%

def train(wrapped_model, data_loader, n_epochs=1):
    pass

n_epochs=1
for epoch in range(n_epochs):
    for batch in tqdm(data_loader):
        print(batch)
        bin_input_data, global_input_data, pla = batch["bin_input_data"], batch["global_input_data"], batch["pla"]
        policy = get_policy(wrapped_model, bin_input_data, global_input_data)
        loss = loss_fn(policy["probs0"], batch["annotated_values"], pla)



train(wrapped_model, data_loader)


# %%

for k, v in policy.items():
    print(k, v.shape)
# %%


sns.scatterplot(y=[r.item() for r in regrets], x=np.arange(52, 152), color='blue')
sns.scatterplot(y=[r.item() for r in regrets_inverted], x=np.arange(52, 152), color='red')
# label this series "regrets":
plt.xlabel("move number")
plt.ylabel("regret")
plt.legend(["normal", "inverted logits"])
plt.show()
# %%
