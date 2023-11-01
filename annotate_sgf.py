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
from typing import List
from einops import repeat
import cProfile
import pstats
sys.path.append("./KataGo/python")

# from analysis engine code
import argparse
import json
import subprocess
import time
from threading import Thread
import sgfmill
import sgfmill.boards
import sgfmill.ascii_boards
from typing import Tuple, List, Optional, Union, Literal
import io

Color = Union[Literal["b"],Literal["w"]]
Move = Union[Literal["pass"],Tuple[int,int]]

from sgfmill import sgf
from KataGo.python.board import Board
from KataGo.python.data import Metadata, load_sgf_moves_exn
# from KataGo.python.play import get_outputs
from KataGo.python.load_model import load_model
from KataGo.python.features import Features
from tqdm import tqdm


# %%

SGF_DIR = 'sgf_downloads'
ANNOTATION_DIR = 'annotations'
TRAINING_DIR = 'training_data'
CHECKPOINT_FILE = 'kg_checkpoint/kata1-b18c384nbt-s7709731328-d3715293823/model.ckpt'
DEVICE = 'cuda'
pos_len = 19

model, swa_model, _ = load_model(CHECKPOINT_FILE, None, device=DEVICE, pos_len=pos_len, verbose=True)
model.eval()
model_config = model.config
if swa_model is not None:
    model = swa_model.module
    model.eval()

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


def sgfmill_to_str(move: Move) -> str:
    if move == "pass":
        return "pass"
    (y,x) = move
    return "ABCDEFGHJKLMNOPQRSTUVWXYZ"[x] + str(y+1)

class KataGoAnnotator:

    def __init__(self, katago_path: str, config_path: str, model_path: str, filename: str, verbose=False):
        self.query_counter = 0
        # open file to write stdout to
        self.out_file = open(filename, 'w')
        katago = subprocess.Popen(
            [katago_path, "analysis", "-config", config_path, "-model", model_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            # bufsize=1,
        )
        self.katago = katago
        self.verbose = verbose

        def printforever():
            while katago.poll() is None:
                data = katago.stderr.readline()
                # time.sleep(0)
                if data and verbose:
                    print("KataGo: ", data.decode(), end="")
            data = katago.stderr.read()
            if data:
                print("KataGo: ", data.decode(), end="")
        self.stderrthread = Thread(target=printforever)
        self.stderrthread.start()

        self.n_writes = 0
        def write_to_outfile():
            pbar = tqdm()
            while katago.poll() is None and not self.katago.stdout.closed:
                data = katago.stdout.readline()
                # time.sleep(0)
                if data:
                    pbar.update(1)
                    self.n_writes += 1
                    # print(f"Writing {self.n_writes}")
                    line = data.decode().strip()
                    json_object = json.loads(line)
                    out_object = {}
                    id = json_object['id'].split()
                    out_object['id'] = id[0]
                    out_object['pla'] = id[1]
                    out_object['loc'] = id[2] + ' ' + id[3]
                    out_object['score'] = json_object['rootInfo']['scoreLead']
                    out_object['turnNumber'] = json_object['turnNumber']
                    out_object['winrate'] = json_object['rootInfo']['winrate']
                    self.out_file.write(json.dumps(out_object) + '\n')
            print(f"Done writing {self.n_writes} lines to {filename}")
            if self.katago.stdout.closed: return
            data = katago.stdout.read()
            if data:
                self.out_file.write(data.decode())
        self.stdoutthread = Thread(target=write_to_outfile)
        self.stdoutthread.start()


    def close(self):
        self.katago.stdin.close()
        self.stdoutthread.join()
        self.out_file.close()
        return self.n_writes

    def query(self, initial_board: sgfmill.boards.Board, moves: List[Tuple[Color,Move]], komi: float, max_visits=None):
        query = {}
        query["id"] = f"{self.query_counter} {moves[-1][0]} {moves[-1][1][0]} {moves[-1][1][1]}"
        self.query_counter += 1

        query["moves"] = [(color,sgfmill_to_str(move)) for color, move in moves]
        query["initialStones"] = []
        # for y in range(initial_board.side):
        #     for x in range(initial_board.side):
        #         color = initial_board.get(y,x)
        #         if color:
        #             query["initialStones"].append((color,sgfmill_to_str((y,x))))
        query["rules"] = "Chinese"
        query["komi"] = komi
        query["boardXSize"] = initial_board.side
        query["boardYSize"] = initial_board.side
        query["includePolicy"] = False
        if max_visits is not None:
            query["maxVisits"] = max_visits
        query = (json.dumps(query) + "\n").encode()
        # print(f"{len(query)=}")
        # time.sleep(0.001)
        self.katago.stdin.write(query)
        self.katago.stdin.flush()

# %%

def loc_to_query_loc(board, loc):
    x, y = board.loc_x(loc), board.loc_y(loc)
    return sgfmill_to_str((y,x)) # am I switching coordinates here?
# %%

KATAGO_PATH = "/home/ubuntu/katago_pessimize/kg_release/katago"
CONFIG_PATH = "/home/ubuntu/katago_pessimize/analysis.cfg"
MODEL_PATH = "/home/ubuntu/katago_pessimize/kata1-b18c384nbt-s7709731328-d3715293823.bin.gz"

def get_query_move(color, loc, board):
    return ('_bw'[color], (board.loc_x(loc), board.loc_y(loc)) if loc != Board.PASS_LOC else "pass")

def get_training_data_from_sgf(katago:KataGoAnnotator, sgf_filename, min_move=50, max_move=150, verbose=False):
    metadata, setup, moves, rules = load_sgf_moves_exn(sgf_filename)
    print(f"Loaded {sgf_filename}")
    print(f"Metadata: {metadata}")
    print(f"Rules: {rules}")
    print(f"Setup: {setup}")
    if verbose: print(f"Moves: {moves}")

    # katago = KataGo(args["katago_path"], args["config_path"], args["model_path"])
    query_board = sgfmill.boards.Board(19)
    komi = 6.5

    gs = GameState(metadata.size)
    # for move in setup:
    #     gs.play(move[0], move[1])
    # print(gs.board.to_string())
    n_queries = 0
    start_time = time.time()
    # queries = 0
    for i, game_move in tqdm(enumerate(moves[:max_move])):
        # print(f"playing {game_move}")
        gs.play(game_move[0], game_move[1])
        board_str = '\n' + gs.board.to_string().strip()
        # print(board_str)
        if i < min_move: continue

        # make all possible moves from this position
        this_values = dict()
        query_moves = [get_query_move(color, loc, gs.board) for color, loc in gs.moves]
        for move_x in range(metadata.size):
            for move_y in range(metadata.size):
                if gs.board.would_be_legal(gs.board.pla, Board.loc_static(move_x, move_y, metadata.size)):
                    n_queries += 1
                    katago.query(query_board, query_moves + [('_bw'[gs.board.pla], (move_x, move_y))], komi)

    end_time = time.time()
    print(f"Querying finished in time: {end_time - start_time:.4f}. Now waiting for gpu...")
    # print(values)
    return gs, n_queries

# %%

def annotate_all_games(overwrite=False, max_games=None, verbose=False):
    files = os.listdir(SGF_DIR)
    if max_games is not None:
        files = files[:max_games]
    for sgf_filename in files:
        print(f"Starting to annotate game {sgf_filename}")
        sgf_rel_path = os.path.join(SGF_DIR, sgf_filename)
        # sgf_filename = "blood_vomit.sgf"
        annotation_rel_path = os.path.join(ANNOTATION_DIR, sgf_filename + '.stdout')
        if os.path.exists(annotation_rel_path) and not overwrite:
            print(f"Skipping {sgf_filename}")
            continue
        katago = KataGoAnnotator(KATAGO_PATH, CONFIG_PATH, MODEL_PATH, annotation_rel_path, verbose=verbose)
        try:
            # cProfile.run('values, move_values, gs = get_training_data_from_sgf(katago, sgf_rel_path)', 'output.pstats')
            gs, n_queries = get_training_data_from_sgf(katago, sgf_rel_path, verbose=verbose)
            n_writes = katago.close()
            if n_queries != n_writes:   
                print(f"Warning: {n_queries} queries but {n_writes} writes on file {sgf_filename}")
        except KeyboardInterrupt:
            n_writes = katago.close()
            raise KeyboardInterrupt
        except Exception as e:
            print(f"Exception: {e}")
            n_writes = katago.close()
        
annotate_all_games(overwrite=True, verbose=False, max_games=1000)

# %%

# Create a pstats object
# p = pstats.Stats('output.pstats')

# Sort the statistics by the cumulative time and print the first few lines
# p.sort_stats('cumulative').print_stats(20)

# %%
