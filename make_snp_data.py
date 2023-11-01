# %%
"""
Converts the SGF files and annotations created with `annotate_sgf.py` into a training set that can be passed to the model,
which has features for a position, and the value for each possible move.

The bottleneck is `fill_row_features`, which can be sped up by using c++ code instead, but I haven't bothered for now.
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
import transformer_lens as tl
import json
from multiprocessing import Pool
from functools import partial
import cProfile
import pstats

from snp_utils import HookedKataGoWrapper
sys.path.append("./KataGo/python")

from sgfmill import sgf
from KataGo.python.board import Board
from KataGo.python.data import Metadata, load_sgf_moves_exn
# from KataGo.python.play import get_outputs
from KataGo.python.load_model import load_model
from KataGo.python.features import Features
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map


# %%

SGF_DIR = 'sgf_downloads'
TRAINING_DIR = 'training_data'
ANNOTATIONS_DIR = 'annotations'
DATASET_DIR = 'dataset'
CHECKPOINT_FILE = 'kg_checkpoint/kata1-b18c384nbt-s7709731328-d3715293823/model.ckpt'
DEVICE = 'cuda'
N_GAMES_IN_DATASET = 3
pos_len = 19

# We load the model just to get bin_input_shape and global_input_shape
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

# %%

def load_annotations(sgf_filename, size=19, start_move=52, end_move=152):
    # Loads everything into a tensor
    annotations_relpath = os.path.join(ANNOTATIONS_DIR, sgf_filename + ".stdout")
    with open(annotations_relpath, 'r') as f:
        annotations = f.read()
    annotations = annotations.split('\n')
    # print(end_move - start_move)
    result = torch.zeros((end_move - start_move, Board(size).arrsize)) + torch.nan
    for line in annotations:
        if not line: continue
        # print(line)
        annotation = json.loads(line)
        loc = Board.loc_static(*map(int,annotation['loc'].split()), size)
        # print(loc)
        result[annotation['turnNumber'] - start_move, loc] = annotation['winrate']
    # print(result.isnan().float().mean())
    return result

# %%
# %%
features = Features(model_config, pos_len)

# copied from play
def get_input_data(gs:GameState, rules, cache=True):
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

        return bin_input_data, global_input_data

# %%

# Projected size: 4*22*19*19*100*1000

def process_file(annotation_filename, start_move=52, end_move=152, overwrite=None):
    # if overwrite is None:
    #     overwrite = '-overwrite' in sys.argv[1:]
    sgf_filename = annotation_filename[:-7]
    dataset_filename = sgf_filename + '.npz'
    dataset_relpath = os.path.join(DATASET_DIR, dataset_filename)
    if os.path.exists(dataset_relpath) and not overwrite:
        print(f"Skipping {sgf_filename}")
        return
    # TODO fix komi and rules when loading?
    metadata, setup, moves, rules = load_sgf_moves_exn(os.path.join(SGF_DIR, sgf_filename))
    # print(f"Loaded {sgf_filename}")
    # print(f"Metadata: {metadata}")
    # print(f"Rules: {rules}")
    # print(f"Setup: {setup}")
    # print(f"Moves: {moves}")

    gs = GameState(metadata.size)

    annotated_values = load_annotations(sgf_filename, size=metadata.size, start_move=start_move, end_move=end_move)
    # for move in setup:
    #     gs.play(move[0], move[1])
    # print(gs.board.to_string())

    
    bin_input_data = np.zeros(shape=[end_move - start_move]+model.bin_input_shape, dtype=np.float32)
    global_input_data = np.zeros(shape=[end_move - start_move]+model.global_input_shape, dtype=np.float32)
    pla = np.zeros(shape=[end_move - start_move], dtype=np.float32)

    for move_n, game_move in enumerate(moves[:end_move]):
        # print(f"playing {game_move}")
        gs.play(game_move[0], game_move[1])
        board_str = '\n' + gs.board.to_string().strip()
        # print(board_str)
        if move_n >= start_move:
            bin_input_data[move_n - start_move], global_input_data[move_n - start_move] = get_input_data(gs, rules)
            pla[move_n - start_move] = gs.board.pla

    zero_mask = pla == 0
    if zero_mask.any():
        print(f"Skipping {sgf_filename} because it's incomplete")
        return

    # print(policies)
    np.savez(os.path.join(DATASET_DIR, sgf_filename + '.npz'), bin_input_data=bin_input_data, global_input_data=global_input_data, annotated_values=annotated_values, pla=pla)

def make_dataset(overwrite=False):
    if overwrite:
        print("Overwriting dataset")
    files_to_process = os.listdir(ANNOTATIONS_DIR)[:N_GAMES_IN_DATASET]
    # with Pool(8) as p, tqdm(total=N_GAMES_IN_DATASET) as pbar:
        # Using partial to set overwrite for all calls
    for file in files_to_process:
        process_file(file, overwrite=overwrite)
        # p.apply_async(process_file, (file,), {'overwrite': overwrite}, callback=lambda _: pbar.update(1))
        
# %%

if __name__ == "__main__":
    args = sys.argv[1:]
    cProfile.run("make_dataset(overwrite=True)", 'output.pstats')


# %%

# Create a pstats object
p = pstats.Stats('output.pstats')

# Sort the statistics by the cumulative time and print the first few lines
p.sort_stats('tottime').print_stats(20)
# %%
