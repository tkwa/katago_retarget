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
    def __init__(self, mod:torch.nn.Module, name='model', recursive=False):
        super().__init__()
        self.mod = deepcopy(mod)
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

    def forward(self, *args, **kwrags):
       result = self.mod.forward(*args, **kwrags)
       return self.hook_point(result)
   
tl_model = HookedModuleWrapper(model, recursive=True)

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
sgf_filename = os.listdir(SGF_DIR)[2]
annotated_values = load_annotations(sgf_filename)


# %%

# copied from play
def get_outputs(gss:GameState | List[GameState], rules, cache=True):
    if type(gss) is GameState:
        gss = [gss]
    with torch.no_grad():
        model.eval()

        batch_size = len(gss)
        bin_input_data = np.zeros(shape=[batch_size]+model.bin_input_shape, dtype=np.float32)
        global_input_data = np.zeros(shape=[1]+model.global_input_shape, dtype=np.float32)
        pla = [gs.board.pla for gs in gss]
        opp = [Board.get_opp(p) for p in pla]
        move_idx = [len(gs.moves) for gs in gss]
        # This function assumes N(HW)C order but we actually use NCHW order, so work with it and revert
        bin_input_data = np.transpose(bin_input_data,axes=(0,2,3,1))
        bin_input_data = bin_input_data.reshape([batch_size,1,pos_len*pos_len,-1])
        featureses = [Features(model.config,pos_len) for i in range(batch_size)]
        for i, gs in enumerate(gss):
            featureses[i].fill_row_features(gs.board,pla,opp,gs.boards,gs.moves,move_idx[i],rules,bin_input_data[i],global_input_data,idx=0)
        bin_input_data = bin_input_data.reshape([batch_size,pos_len,pos_len,-1])
        bin_input_data = np.transpose(bin_input_data,axes=(0,3,1,2))

        # Currently we don't actually do any symmetries
        # symmetry = 0
        # model_outputs = model(apply_symmetry(batch["binaryInputNCHW"],symmetry),batch["globalInputNC"])

        if cache:
            model_outputs, cache = tl_model.run_with_cache(
                torch.tensor(bin_input_data, dtype=torch.float32).to(DEVICE),
                torch.tensor(global_input_data, dtype=torch.float32).to(DEVICE),
            )
            for k, v in cache.items():
                print(f"cache {k} {v.shape} {v.mean()}")
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

        policy0 = torch.nn.functional.softmax(policy_logits[0,:],dim=0).cpu().numpy()
        policy1 = torch.nn.functional.softmax(policy_logits[1,:],dim=0).cpu().numpy()
        value = torch.nn.functional.softmax(value_logits,dim=0).cpu().numpy()
        td_value = torch.nn.functional.softmax(td_value_logits[0,:],dim=0).cpu().numpy()
        td_value2 = torch.nn.functional.softmax(td_value_logits[1,:],dim=0).cpu().numpy()
        td_value3 = torch.nn.functional.softmax(td_value_logits[2,:],dim=0).cpu().numpy()
        scoremean = pred_scoremean.cpu().item()
        td_score = pred_td_score.cpu().numpy()
        scorestdev = pred_scorestdev.cpu().item()
        lead = pred_lead.cpu().item()
        vtime = pred_variance_time.cpu().item()
        estv = math.sqrt(pred_shortterm_value_error.cpu().item())
        ests = math.sqrt(pred_shortterm_score_error.cpu().item())
        ownership = torch.tanh(ownership_pretanh).cpu().numpy()
        scoring = pred_scoring.cpu().numpy()
        futurepos = torch.tanh(futurepos_pretanh).cpu().numpy()
        seki_probs = torch.nn.functional.softmax(seki_logits[0:3,:,:],dim=0).cpu().numpy()
        seki = seki_probs[1] - seki_probs[2]
        seki2 = torch.sigmoid(seki_logits[3,:,:]).cpu().numpy()
        scorebelief = torch.nn.functional.softmax(scorebelief_logits,dim=0).cpu().numpy()

    # for gs, features in zip(gss, featureses):
    #     board = gs.board
    #     moves_and_probs0 = []
    #     for i in range(len(policy0)):
    #         move = features.tensor_pos_to_loc(i,board)
    #         if i == len(policy0)-1:
    #             moves_and_probs0.append((Board.PASS_LOC,policy0[i]))
    #         elif board.would_be_legal(board.pla,move):
    #             moves_and_probs0.append((move,policy0[i]))

    #     moves_and_probs1 = []
    #     for i in range(len(policy1)):
    #         move = features.tensor_pos_to_loc(i,board)
    #         if i == len(policy1)-1:
    #             moves_and_probs1.append((Board.PASS_LOC,policy1[i]))
    #         elif board.would_be_legal(board.pla,move):
    #             moves_and_probs1.append((move,policy1[i]))

    #     ownership_flat = ownership.reshape([features.pos_len * features.pos_len])
    #     ownership_by_loc = []
    #     board = gss.board
    #     for y in range(board.size):
    #         for x in range(board.size):
    #             loc = board.loc(x,y)
    #             pos = features.loc_to_tensor_pos(loc,board)
    #             if board.pla == Board.WHITE:
    #                 ownership_by_loc.append((loc,ownership_flat[pos]))
    #             else:
    #                 ownership_by_loc.append((loc,-ownership_flat[pos]))

    #     scoring_flat = scoring.reshape([features.pos_len * features.pos_len])
    #     scoring_by_loc = []
    #     board = gss.board
    #     for y in range(board.size):
    #         for x in range(board.size):
    #             loc = board.loc(x,y)
    #             pos = features.loc_to_tensor_pos(loc,board)
    #             if board.pla == Board.WHITE:
    #                 scoring_by_loc.append((loc,scoring_flat[pos]))
    #             else:
    #                 scoring_by_loc.append((loc,-scoring_flat[pos]))

    #     futurepos0_flat = futurepos[0,:,:].reshape([features.pos_len * features.pos_len])
    #     futurepos0_by_loc = []
    #     board = gss.board
    #     for y in range(board.size):
    #         for x in range(board.size):
    #             loc = board.loc(x,y)
    #             pos = features.loc_to_tensor_pos(loc,board)
    #             if board.pla == Board.WHITE:
    #                 futurepos0_by_loc.append((loc,futurepos0_flat[pos]))
    #             else:
    #                 futurepos0_by_loc.append((loc,-futurepos0_flat[pos]))

    #     futurepos1_flat = futurepos[1,:,:].reshape([features.pos_len * features.pos_len])
    #     futurepos1_by_loc = []
    #     board = gss.board
    #     for y in range(board.size):
    #         for x in range(board.size):
    #             loc = board.loc(x,y)
    #             pos = features.loc_to_tensor_pos(loc,board)
    #             if board.pla == Board.WHITE:
    #                 futurepos1_by_loc.append((loc,futurepos1_flat[pos]))
    #             else:
    #                 futurepos1_by_loc.append((loc,-futurepos1_flat[pos]))

    #     seki_flat = seki.reshape([features.pos_len * features.pos_len])
    #     seki_by_loc = []
    #     board = gss.board
    #     for y in range(board.size):
    #         for x in range(board.size):
    #             loc = board.loc(x,y)
    #             pos = features.loc_to_tensor_pos(loc,board)
    #             if board.pla == Board.WHITE:
    #                 seki_by_loc.append((loc,seki_flat[pos]))
    #             else:
    #                 seki_by_loc.append((loc,-seki_flat[pos]))

    #     seki_flat2 = seki2.reshape([features.pos_len * features.pos_len])
    #     seki_by_loc2 = []
    #     board = gss.board
    #     for y in range(board.size):
    #         for x in range(board.size):
    #             loc = board.loc(x,y)
    #             pos = features.loc_to_tensor_pos(loc,board)
    #             seki_by_loc2.append((loc,seki_flat2[pos]))

    #     moves_and_probs = sorted(moves_and_probs0, key=lambda moveandprob: moveandprob[1], reverse=True)
    #     # Generate a random number biased small and then find the appropriate move to make
    #     # Interpolate from moving uniformly to choosing from the triangular distribution
    #     alpha = 1
    #     beta = 1 + math.sqrt(max(0,len(gss.moves)-20))
    #     r = np.random.beta(alpha,beta)
    #     probsum = 0.0
    #     i = 0
    #     genmove_result = Board.PASS_LOC
    #     while True:
    #         (move,prob) = moves_and_probs[i]
    #         probsum += prob
    #         if i >= len(moves_and_probs)-1 or probsum > r:
    #             genmove_result = move
    #             break
    #         i += 1

        return {
            "policy0": policy0,
            "policy1": policy1,
            # "moves_and_probs0": moves_and_probs0,
            # "moves_and_probs1": moves_and_probs1,
            "value": value,
            "td_value": td_value,
            "td_value2": td_value2,
            "td_value3": td_value3,
            "scoremean": scoremean,
            "td_score": td_score,
            "scorestdev": scorestdev,
            "lead": lead,
            "vtime": vtime,
            "estv": estv,
            "ests": ests,
            "ownership": ownership,
            # "ownership_by_loc": ownership_by_loc,
            "scoring": scoring,
            # "scoring_by_loc": scoring_by_loc,
            "futurepos": futurepos,
            # "futurepos0_by_loc": futurepos0_by_loc,
            # "futurepos1_by_loc": futurepos1_by_loc,
            "seki": seki,
            # "seki_by_loc": seki_by_loc,
            "seki2": seki2,
            # "seki_by_loc2": seki_by_loc2,
            "scorebelief": scorebelief,
            # "genmove_result": genmove_result
        }

# %%

def get_training_data_from_sgf(sgf_filename):
    global rules
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
    values = []
    move_values = []
    for game_move in tqdm(moves[:5]):
        print(f"playing {game_move}")
        gs.play(game_move[0], game_move[1])
        board_str = '\n' + gs.board.to_string().strip()
        # print(board_str)

        outputs = get_outputs(gs, rules)
        values.append(outputs["value"][0] + outputs["value"][2]/2)

        # make all possible moves from this position
        this_values = dict()
        for move in range(5): # range(metadata.size ** 2):
            if gs.board.would_be_legal(gs.board.pla, move):
                gs_copy = deepcopy(gs)
                gs_copy.play(gs.board.pla, move)
                outputs = get_outputs(gs_copy, rules)
                this_values[move] = outputs["value"][0] + outputs["value"][2]/2
        move_values.append(this_values)

    print(values)
    return values, gs


# get first file in SGF_DIR
sgf_filename = os.listdir(SGF_DIR)[2]
sgf_relpath = os.path.join(SGF_DIR, sgf_filename)
# sgf_filename = "blood_vomit.sgf"
values, gs = get_training_data_from_sgf(sgf_relpath)

print(gs.board.to_string())

# %%

sns.lineplot(data=values[::2])
# %%
