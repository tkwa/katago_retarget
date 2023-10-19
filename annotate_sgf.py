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
sys.path.append("/home/ubuntu/katago_pessimize/KataGo/python")

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

class KataGo:

    def __init__(self, katago_path: str, config_path: str, model_path: str):
        self.query_counter = 0
        katago = subprocess.Popen(
            [katago_path, "analysis", "-config", config_path, "-model", model_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self.katago = katago

        def printforever():
            while katago.poll() is None:
                data = katago.stderr.readline()
                # time.sleep(0)
                if data:
                    print("KataGo: ", data.decode(), end="")
            data = katago.stderr.read()
            if data:
                print("KataGo: ", data.decode(), end="")
        self.stderrthread = Thread(target=printforever)
        self.stderrthread.start()

    def close(self):
        self.katago.stdin.close()


    def query(self, initial_board: sgfmill.boards.Board, moves: List[Tuple[Color,Move]], komi: float, max_visits=None):
        query = {}

        query["id"] = str(self.query_counter)
        self.query_counter += 1

        query["moves"] = [(color,sgfmill_to_str(move)) for color, move in moves]
        query["initialStones"] = []
        for y in range(initial_board.side):
            for x in range(initial_board.side):
                color = initial_board.get(y,x)
                if color:
                    query["initialStones"].append((color,sgfmill_to_str((y,x))))
        query["rules"] = "Chinese"
        query["komi"] = komi
        query["boardXSize"] = initial_board.side
        query["boardYSize"] = initial_board.side
        query["includePolicy"] = False
        if max_visits is not None:
            query["maxVisits"] = max_visits

        self.katago.stdin.write((json.dumps(query) + "\n").encode())
        self.katago.stdin.flush()

        # print(json.dumps(query))

    def get_response(self):
        line = ""
        while line == "":
            if self.katago.poll():
                time.sleep(1)
                raise Exception("Unexpected katago exit")
            # time.sleep(0.002)
            line = self.katago.stdout.readline()
            # print(f"Successfully read line {line}")
            line = line.decode().strip()
            # print("Got: " + line)
        response = json.loads(line)
        return response
    
    def get_all_responses(self):
        lines = ''
        if self.katago.poll():
            time.sleep(1)
            raise Exception("Unexpected katago exit")
        lines = self.katago.stdout.readlines()
        if lines == '':
            raise Exception("No line")
        # print(f"Successfully read line {line}")
        lines = [line.decode().strip() for line in lines]
        response = [json.loads(line) for line in lines]
        return response

if __name__ == "__main__":
    # description = """
    # Example script showing how to run KataGo analysis engine and query it from python.
    # """
    # parser = argparse.ArgumentParser(description=description)
    # parser.add_argument(
    #     "-katago-path",
    #     help="Path to katago executable",
    #     required=True,
    # )
    # parser.add_argument(
    #     "-config-path",
    #     help="Path to KataGo analysis config (e.g. cpp/configs/analysis_example.cfg in KataGo repo)",
    #     required=True,
    # )
    # parser.add_argument(
    #     "-model-path",
    #     help="Path to neural network .bin.gz file",
    #     required=True,
    # )
    # args = vars(parser.parse_args())
    # print(args)

    # katago = KataGo(args["katago_path"], args["config_path"], args["model_path"])

    board = sgfmill.boards.Board(19)
    komi = 6.5
    # moves = [("b",(3,3))]

    # displayboard = board.copy()
    # for color, move in moves:
    #     if move != "pass":
    #         row,col = move
    #         displayboard.play(row,col,color)
    # print(sgfmill.ascii_boards.render_board(displayboard))

    # print("Query result: ")
    # for i in range(10000):
    #     q = katago.query(board, moves, komi)

    # katago.close()

# %%

def loc_to_query_loc(board, loc):
    x, y = board.loc_x(loc), board.loc_y(loc)
    return sgfmill_to_str((y,x)) # am I switching coordinates here?
# %%

KATAGO_PATH = "/home/ubuntu/katago_pessimize/kg_release/katago"
CONFIG_PATH = "/home/ubuntu/katago_pessimize/analysis.cfg"
MODEL_PATH = "/home/ubuntu/katago_pessimize/kata1-b18c384nbt-s7709731328-d3715293823.bin.gz"

elapsed_out = []
def get_training_data_from_sgf(katago:KataGo, sgf_filename):
    global rules
    global elapsed_out
    metadata, setup, moves, rules = load_sgf_moves_exn(sgf_filename)
    print(f"Loaded {sgf_filename}")
    print(f"Metadata: {metadata}")
    print(f"Rules: {rules}")
    print(f"Setup: {setup}")
    print(f"Moves: {moves}")

    # katago = KataGo(args["katago_path"], args["config_path"], args["model_path"])
    query_board = sgfmill.boards.Board(19)
    komi = 6.5

    gs = GameState(metadata.size)
    # for move in setup:
    #     gs.play(move[0], move[1])
    # print(gs.board.to_string())
    values = []
    move_values = []
    start_time = time.time()
    queries = 0
    for game_move in tqdm(moves[:5]):
        # print(f"playing {game_move}")
        gs.play(game_move[0], game_move[1])
        board_str = '\n' + gs.board.to_string().strip()
        # print(board_str)

        # make all possible moves from this position
        this_values = dict()
        for move in range(metadata.size ** 2):
            move = move + 19 + 1
            if gs.board.would_be_legal(gs.board.pla, move):
                query_moves = [('_bw'[color], (gs.board.loc_x(loc), gs.board.loc_y(loc))) for color, loc in gs.moves + [(gs.board.pla, move)]]
                # print(query_moves)
                katago.query(query_board, query_moves, komi)
                queries += 1
                # if queries >= 100:
                #     queries = 0
    # outputs = katago.get_all_responses()
    time.sleep(4)
    for _ in range(queries):
        before_out = time.time()
        output = katago.get_response()
        elapsed_out.append(time.time() - before_out)
        if 'rootInfo' in output: values.append(output['rootInfo']["winrate"])
    # for output in outputs:

                # for move in range(metadata.size ** 2):

            
        move_values.append(this_values)

    end_time = time.time()
    print(f"Time: {end_time - start_time:.4f}")
    # print(values)
    return values, move_values, gs


# get first file in SGF_DIR
sgf_filename = os.path.join(SGF_DIR, os.listdir(SGF_DIR)[2])
# sgf_filename = "blood_vomit.sgf"

katago = KataGo(KATAGO_PATH, CONFIG_PATH, MODEL_PATH)
try:
    cProfile.run('values, move_values, gs = get_training_data_from_sgf(katago, sgf_filename)', 'output.pstats')
    # values, move_values, gs = get_training_data_from_sgf(katago, sgf_filename)
    katago.close()
except KeyboardInterrupt:
    katago.close()
    raise KeyboardInterrupt
except Exception as e:
    print(e)
    katago.close()
    raise e

# print(gs.board.to_string())

# %%

sns.lineplot(elapsed_out)
# %%

# Create a pstats object
p = pstats.Stats('output.pstats')

# Sort the statistics by the cumulative time and print the first few lines
p.sort_stats('cumulative').print_stats(20)

# %%

sns.lineplot(data=values[::2])
# %%
