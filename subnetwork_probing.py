# %%
"""
Converts the SGF files into a training set that can be passed to the model,
which has a position and the value for each possible move.
"""

import os
import sys
import numpy as np
import torch
import math
import matplotlib.pyplot as plt
import seaborn as sns
from copy import deepcopy
from typing import List, Union
from torch import Tensor
from torch.nn.modules.module import Module
import transformer_lens as tl
import json
from transformer_lens.hook_points import HookedRootModule, HookPoint
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
TRAINING_DIR = 'training_data'
ANNOTATIONS_DIR = 'annotations'
CHECKPOINT_FILE = 'kg_checkpoint/kata1-b18c384nbt-s7709731328-d3715293823/model.ckpt'
DEVICE = 'cuda'
pos_len = 19

model, swa_model, _ = load_model(CHECKPOINT_FILE, None, device=DEVICE, pos_len=pos_len, verbose=True)
model.eval()
model_config = model.config
if swa_model is not None:
    model = swa_model.module
    model.eval()


# %%
# features = Features(model_config, pos_len)

class GameState:
    def __init__(self,board_size):
        self.board_size = board_size
        self.board = Board(size=board_size)
        self.moves = []
        self.boards = [self.board.copy()]

    def play(self, pla, loc):
        self.board.play(pla, loc)
        self.moves.append((pla, loc))
        self.boards.append(self.board.copy())
        # print(f"Now {len(self.moves)} moves and {len(self.boards)} boards")

class HookedModuleWrapper(HookedRootModule):
    """
    Wraps any module, adding a hook after the output.
    """
    def __init__(self, mod:torch.nn.Module, name='model', recursive=False, hook_self=True):
        super().__init__()
        self.mod = deepcopy(mod)
        self.hook_self = hook_self
        if hook_self:
            hook_point = HookPoint()
            hook_point.name = name
            self.hook_point = hook_point
        if recursive: self.wrap_hookpoints_recursively()
        # TODO set names for all HookedSubmoduleWrappers
        self.setup()

    def wrap_hookpoints_recursively(self):
        for key, submod in list(self.mod._modules.items()):
            if isinstance(submod, HookedModuleWrapper):
                print(f"SKIPPING {key}:{type(submod)}")
                continue
            if key in ['intermediate_value_head', 'value_head']: # these return tuples
                print(f"SKIPPING {key}:{type(submod)}")
                continue
            if isinstance(submod, torch.nn.ModuleList):
                print(f"INDIVIDUALLY WRAPPING {key}:{type(submod)}")
                for i, subsubmod in enumerate(submod):
                    new_submod = HookedModuleWrapper(subsubmod, name=f'{key}.{i}', recursive=True)
                    submod[i] = new_submod
                continue
            # print(f'wrapping {key}:{type(submod)}')
            new_submod = HookedModuleWrapper(submod, name='hook_' + key, recursive=True)
            self.mod.__setattr__(key, new_submod)

    def forward(self, *args, **kwargs):
       result = self.mod.forward(*args, **kwargs)
       if not self.hook_self:
           return result
       assert isinstance(result, Tensor)
       return self.hook_point(result)
   
tl_model = HookedModuleWrapper(model, recursive=True, hook_self=False)

# TODO add tests for HookedModuleWrapper

# %%

def load_annotations(sgf_filename, size=19, start_move=52, end_move=152):
    # Loads everything into a tensor
    annotations_relpath = os.path.join(ANNOTATIONS_DIR, sgf_filename + ".stdout")
    with open(annotations_relpath, 'r') as f:
        annotations = f.read()
    annotations = annotations.split('\n')
    result = torch.zeros((end_move - start_move, Board(size).arrsize)) + torch.nan
    for line in annotations:
        if not line: continue
        print(line)
        annotation = json.loads(line)
        loc = Board.loc_static(*map(int,annotation['loc'].split()), size)
        print(loc)
        result[annotation['turnNumber'] - start_move, loc] = annotation['winrate']
    return result

# %%
# %%
features = Features(model_config, pos_len)

# copied from play
def get_policy(gs:GameState, rules, cache=True):
    with torch.no_grad():
        model.eval()

        bin_input_data = np.zeros(shape=[1]+model.bin_input_shape, dtype=np.float32)
        global_input_data = np.zeros(shape=[1]+model.global_input_shape, dtype=np.float32)
        pla = gs.board.pla
        opp = Board.get_opp(pla)
        move_idx = len(gs.moves)
        # This function assumes N(HW)C order but we actually use NCHW order, so work with it and revert
        bin_input_data = np.transpose(bin_input_data,axes=(0,2,3,1))
        bin_input_data = bin_input_data.reshape([1,pos_len*pos_len,-1])
        features.fill_row_features(gs.board,pla,opp,gs.boards,gs.moves,move_idx,rules,bin_input_data,global_input_data,idx=0)
        bin_input_data = bin_input_data.reshape([1,pos_len,pos_len,-1])
        bin_input_data = np.transpose(bin_input_data,axes=(0,3,1,2))

        # Currently we don't actually do any symmetries
        # symmetry = 0
        # model_outputs = model(apply_symmetry(batch["binaryInputNCHW"],symmetry),batch["globalInputNC"])

        if cache:
            model_outputs, cache = tl_model.run_with_cache(
                torch.tensor(bin_input_data, dtype=torch.float32).to(DEVICE),
                torch.tensor(global_input_data, dtype=torch.float32).to(DEVICE),
            )
            # for k, v in cache.items():
            #     print(f"cache {k} {v.shape} {v.mean()}")
        else:
            model_outputs = model(
                torch.tensor(bin_input_data, dtype=torch.float32).to(DEVICE),
                torch.tensor(global_input_data, dtype=torch.float32).to(DEVICE),
            )

        outputs = model.postprocess_output(model_outputs)
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
        ) = (x[0] for x in outputs[0]) # N = 0

        policy0 = torch.nn.functional.softmax(policy_logits[0,:],dim=0)
        policy_inverted = torch.nn.functional.softmax(- policy_logits[0,:], dim=0)
        value = torch.nn.functional.softmax(value_logits,dim=0).cpu().numpy()


        board = gs.board

        probs0 = torch.zeros(board.arrsize)
        for i in range(len(policy0)):
            move = features.tensor_pos_to_loc(i,board)
            if i == len(policy0)-1:
                pass
                # probs0[Board.PASS_LOC] = policy0[i]
            elif board.would_be_legal(board.pla,move):
                probs0[move] = policy0[i]

        probs_inverted = torch.zeros(board.arrsize)
        for i in range(len(policy_inverted)):
            move = features.tensor_pos_to_loc(i,board)
            if i == len(policy_inverted)-1:
                pass
                # probs_inverted[Board.PASS_LOC] = policy_inverted[i]
            elif board.would_be_legal(board.pla,move):
                probs_inverted[move] = policy_inverted[i]


        return {
            "policy0": policy0,
            "policy_inverted": policy_inverted,
            # "policy1": policy1,
            "probs0": probs0,
            "probs_inverted": probs_inverted,
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
    assert pla in [1, 2]
    if pla == 2: # all annotated values are black's winrate
        annotated_values = 1 - annotated_values
    # Policy should be 0 wherever annotated value is nan
    nan_mask = torch.isnan(annotated_values)
    if policy_probs[nan_mask].nonzero().shape[0] != 0:
        bad_list = policy_probs[nan_mask].nonzero()
        print(f"Warning: policy_probs has non-zero values where annotated_values is nan, at {bad_list.numel()} locations {bad_list.tolist()}")
        # policy_board = Board(19)
        # policy_board.board = (policy_probs == 0).int()
        # print(f"Policy board:")
        # print(policy_board.to_string())

        # annotated_values_board = Board(19)
        # annotated_values_board.board = annotated_values.isnan()
        # print(f"Values board:")
        # print(annotated_values_board.to_string())

    annotated_values = torch.nan_to_num(annotated_values, nan=torch.inf)
    annotated_values = annotated_values - annotated_values.min() # regrets
    annotated_values[nan_mask] = 0
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

def calc_game_regrets(sgf_filename, annotated_values, start_move=52, end_move=152):
    # TODO fix komi and rules when loading?
    metadata, setup, moves, rules = load_sgf_moves_exn(sgf_filename)
    print(f"Loaded {sgf_filename}")
    print(f"Metadata: {metadata}")
    print(f"Rules: {rules}")
    print(f"Setup: {setup}")
    print(f"Moves: {moves}")

    gs = GameState(metadata.size)
    # for move in setup:
    #     gs.play(move[0], move[1])
    # print(gs.board.to_string())
    policies = []
    regrets = []
    values = []
    regrets_inverted = []
    for move_n, game_move in tqdm(enumerate(moves[:end_move])):
        # print(f"playing {game_move}")
        gs.play(game_move[0], game_move[1])
        board_str = '\n' + gs.board.to_string().strip()
        # print(board_str)
        if move_n >= start_move:
            outputs = get_policy(gs, rules)
            value = outputs['value'][0] + 0.5 * outputs['value'][2]
            values.append(outputs['value'])
            policies.append(outputs["policy0"])
            loss = loss_fn(outputs['probs0'], annotated_values[move_n - start_move], gs.board.pla)
            regrets.append(loss)
            loss_inverted = loss_fn(outputs['probs_inverted'], annotated_values[move_n - start_move], gs.board.pla)
            regrets_inverted.append(loss_inverted)

    # print(policies)
    return policies, values, regrets, regrets_inverted, gs


# get first file in SGF_DIR
sgf_filename = os.listdir(SGF_DIR)[345]
sgf_relpath = os.path.join(SGF_DIR, sgf_filename)
# sgf_filename = "blood_vomit.sgf"
annotated_values = load_annotations(sgf_filename)
policies, values, regrets, regrets_inverted, gs = calc_game_regrets(sgf_relpath, annotated_values)

print(gs.board.to_string())

print(regrets)

# %%

sns.scatterplot(y=[r.item() for r in regrets], x=np.arange(52, 152), color='blue')
sns.scatterplot(y=[r.item() for r in regrets_inverted], x=np.arange(52, 152), color='red')
# label this series "regrets":
plt.xlabel("move number")
plt.ylabel("regret")
plt.legend(["normal", "inverted logits"])
plt.show()
# %%
