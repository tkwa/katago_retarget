# %%
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader
import time
import wandb
import argparse
import gc
from typing import Optional, Dict, Tuple, Union
import matplotlib
import matplotlib.pyplot as plt
from IPython.display import SVG
from tqdm import tqdm
# import cProfile
# import pstats
# from torch.profiler import profile, record_function, ProfilerActivity
sys.path.append("..")
sys.path.append("../KataGo/python")
from snp_utils import HookedKataGoWrapper, HookedKataTrainingObject, KataPessimizeDataset, mask_flippedness

# from KataGo.python.play import get_outputs
from KataGo.python.load_model import load_model
from KataGo.python.model_pytorch import Model


from KataGo.python.board import Board

# %%

RULES = {
    "koRule": "KO_POSITIONAL",
    "scoringRule": "SCORING_AREA",
    "taxRule": "TAX_NONE",
    "multiStoneSuicideLegal": True,
    "hasButton": False,
    "encorePhase": 0,
    "passWouldEndPhase": False,
    "whiteKomi": 0.5,
}

class GameState:
    """
    Class copied from nb_common.py
    """
    def __init__(self, board_size, board=None, rules=None, model=None, model_outputs=None):
        self.board_size = board_size
        if board is None:
            board = Board(size=board_size)
        self.board = board
        assert self.board_size == self.board.size

        self.moves = []
        self.boards = [self.board.copy()]
        self.rules = rules if rules is not None else RULES
        self.model = model
        self.model_outputs = model_outputs

    def copy(self):
        gs = GameState(self.board_size, board=self.board.copy(), rules=self.rules.copy())
        gs.moves = self.moves.copy()
        gs.boards = self.boards.copy()
        return gs

    def feed_dict(self, model=None):
        model = model or self.model
        assert model is not None

        bin_input_data = np.zeros(shape=[1] + model.bin_input_shape, dtype=np.float32)
        global_input_data = np.zeros(shape=[1] + model.global_input_shape, dtype=np.float32)
        pla = self.board.pla
        opp = Board.get_opp(pla)
        move_idx = len(self.moves)
        model.fill_row_features(
            self.board,
            pla,
            opp,
            self.boards,
            self.moves,
            move_idx,
            RULES,
            bin_input_data,
            global_input_data,
            idx=0,
        )
        feed_dict = {
            model.bin_inputs: bin_input_data,
            model.global_inputs: global_input_data,
            model.symmetries: np.zeros([3], dtype=np.bool),
            model.include_history: np.ones([1, 5], dtype=np.float32),
        }
        return feed_dict

    def play(self, player: int, move_xy: Tuple[int, int]):
        move = self.board.loc(*move_xy)
        self.board.play(player, move)
        self.moves.append((player, move))
        self.boards.append(self.board.copy())

    def board_as_square(self):
        return self.board.board[:-1].reshape((self.board_size + 2, self.board_size + 1))[1:-1, 1:]

    def show(
        self,
        policy: Optional[np.ndarray] = None,
        w: float = 20,
        heatmap_min: Optional[float] = None,
        heatmap_max: Optional[float] = None,
        cmap: Optional[Union[matplotlib.colors.ListedColormap, matplotlib.colors.Colormap]] = None,
    ):
        """Visualize currently GoBoard state with an optional policy heatmap."""
        n = self.board_size
        if cmap is None:
            cmap = plt.get_cmap("viridis")
        extra = []

        def to_xy(cell):
            """Maps a cell (col, row) to an SVG (x, y) coordinate"""
            return w + w * cell[0], w + w * cell[1]

        # Draw the heatmap, if present.
        if policy is not None:
            if policy.shape[:2] == (n, n):
                heatmap = policy
            else:
                heatmap = policy[:-1].reshape((n, n))
            assert heatmap.shape[:2] == (n, n)
            if heatmap_min is None:
                heatmap_min = heatmap.min()
            if heatmap_max is None:
                heatmap_max = heatmap.max() + 1e-20
            assert heatmap_min is not None and heatmap_max is not None
            normalized = (heatmap - heatmap_min) / (heatmap_max - heatmap_min)
            for cell_y in range(n):
                for cell_x in range(n):
                    value = normalized[cell_y, cell_x]
                    x, y = to_xy((cell_x, cell_y))
                    if len(value.shape) == 0:
                        r, g, b, _ = cmap(value)
                    else:
                        r, g, b, *_ = value
                    color = f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"

                    extra.append(
                        f"""
                        <rect
                        x="{x - w/2 - 0.25}" y="{y - w/2 - 0.25}"
                        width="{w + 0.5}" height="{w + 0.5}"
                        style="fill: {color}; stroke: none; opacity: 1.0;"
                        />
                    """
                    )
        # Draw and label the grid.
        for i in range(n):
            extra.append(
                f"""
                <line x1="{w + w * i}" x2="{w + w * i}" y1="{w}" y2="{w * 20 - w}" stroke="black" />
                <line y1="{w + w * i}" y2="{w + w * i}" x1="{w}" x2="{w * 20 - w}" stroke="black" />
                <text x="{w - 15}" y="{w + w * i + 5}" font="bold 2px" style="fill: #ffe7a3; stroke: none;">{str(i)}</text>
                <text x="{w + w * i - 5}" y="{w - 5}" font="bold 2px" style="fill: #ffe7a3; stroke: none;">{str(i)}</text>
            """
            )
        # Draw the star points.
        for i in range(3):
            for j in range(3):
                hoshi = 3 + 6 * i, 3 + 6 * j
                xy = to_xy(hoshi)
                extra.append(
                    f"""
                    <circle cx="{xy[0]}" cy="{xy[1]}" r="4" fill="black" />
                """
                )
        # Render all of the stones on the board.
        for i in range(n):
            for j in range(n):
                xy = to_xy((i, j))
                stone = self.board.board[self.board.loc(i, j)]
                if stone == Board.BLACK:
                    fill = "#000"
                    stroke = "#fff"
                elif stone == Board.WHITE:
                    fill = "#fff"
                    stroke = "#000"
                else:
                    continue
                extra.append(
                    f"""
                    <circle cx="{xy[0]}" cy="{xy[1]}" r="{(w - 2)/2}" style="fill: {fill}; stroke-width: 1; stroke: {stroke};" />
                """
                )

        # Display the SVG.
        content = f"""
        <!-- Board -->
        <rect
            width="{w * 20}" height="{w * 20}"
            rx="{0.75 * w}"
            style="fill: #966f33; stroke-width: 2; stroke: black;"
        />
        {"".join(extra)}
        """
        svg = f"""
        <svg height="{w * 20}" width="{w * 20}" viewBox="-5 -5 {w * 20 + 10} {w * 20 + 10}" version="1.1">
        {content}
        </svg>
        """
        return SVG(svg)

# %%

DATASET_DIR = '../dataset'
HOOKED_MODEL_FILE = '../models/model_34_20231102-063129.pth'
CHECKPOINT_FILE = '../kg_checkpoint/kata1-b18c384nbt-s7709731328-d3715293823/model.ckpt'
DEVICE = 'cuda'
pos_len = 19

kata_model, swa_model, _ = load_model(CHECKPOINT_FILE, None, device=DEVICE, pos_len=pos_len, verbose=True)

# %%
state_dict = torch.load(HOOKED_MODEL_FILE,map_location="cpu")
if "config" in state_dict:
    model_config = state_dict["config"]
    del state_dict["config"]
else:
    model_config = kata_model.config
inner_model = Model(model_config,pos_len)
inner_model.initialize()
wrapped_model = HookedKataGoWrapper(inner_model).to(DEVICE)
wrapped_model.load_state_dict(state_dict)
# %%

dataset = KataPessimizeDataset(DATASET_DIR, n_games=10)
val_frac = 0.1
_, val_set = torch.utils.data.random_split(dataset, [int(len(dataset)*(1-val_frac)), int(len(dataset)*val_frac)])

# %%

training_object = HookedKataTrainingObject(model_config, DEVICE)

# %%
# loader = DataLoader(val_set, batch_size=6, shuffle=True, num_workers=1)

def get_board(data_point, size=19):
    bin_input_data = data_point["bin_input_data"]
    pla = int(data_point["pla"])
    pla_stones, opp_stones = bin_input_data[1], bin_input_data[2]
    board = Board(size=pos_len)
    for x in range(size):
        for y in range(size):
            if pla_stones[x, y]:
                board.play(pla, board.loc(x, y))
            elif opp_stones[x, y]:
                board.play(Board.get_opp(pla), board.loc(x, y))
    return board

data_point = val_set[1]
board = get_board(data_point)
gs = GameState(board_size=19, board=board, model=wrapped_model)
bin_input_data = torch.tensor(data_point["bin_input_data"]).unsqueeze(0) # add batch dim
global_input_data = torch.tensor(data_point["global_input_data"]).unsqueeze(0)
annotated_values = torch.tensor(data_point["annotated_values"]).unsqueeze(0) # add batch dim
policy = training_object.get_model_outputs(
    wrapped_model, bin_input_data, global_input_data, annotated_values)['policy0']

policy = policy[:, :-1].reshape((19, 19)).detach().cpu().numpy()
# %%
print(f"player is {'BLACK' if gs.board.pla == 1 else 'WHITE'}")
gs.show(policy)
# %%
