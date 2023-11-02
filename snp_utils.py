# %%
from copy import deepcopy
import os
import sys
import math
from einops import repeat
import numpy as np
import torch
from torch import Tensor
from typing import List, Dict, Tuple, Callable, ContextManager, Self
from torch.utils.data import Dataset
from transformer_lens.hook_points import HookPoint, HookedRootModule
from KataGo.python.board import Board
from KataGo.python.features import Features

sys.path.append("./KataGo/python")
from KataGo.python.model_pytorch import Model as KataModel
# %%

# SGF_DIR = 'sgf_downloads'
# TRAINING_DIR = 'training_data'
# ANNOTATIONS_DIR = 'annotations'
# CHECKPOINT_FILE = 'kg_checkpoint/kata1-b18c384nbt-s7709731328-d3715293823/model.ckpt'
# DEVICE = 'cuda'
# pos_len = 19

# model, swa_model, _ = load_model(CHECKPOINT_FILE, None, device=DEVICE, pos_len=pos_len, verbose=True)
# model.eval()
# model_config = model.config
# if swa_model is not None:
#     model = swa_model.module
#     model.eval()
# %%

class HookedModuleWrapper(HookedRootModule):
    """
    Wraps any module, adding a hook after the output.
    """
    def __init__(self, mod:torch.nn.Module, name='model', recursive=False, hook_self=True, top_level=True):
        super().__init__()
        self.mod = mod # deepcopy(mod)
        self.hook_self = hook_self
        if hook_self:
            hook_point = HookPoint()
            hook_point.name = name
            self.hook_point = hook_point
        if recursive: self.wrap_hookpoints_recursively()
        self.setup()

    def wrap_hookpoints_recursively(self, verbose=False):
        show = lambda *args: print(*args) if verbose else None
        for key, submod in list(self.mod._modules.items()):
            if isinstance(submod, HookedModuleWrapper):
                show(f"SKIPPING {key}:{type(submod)}")
                continue
            if key in ['intermediate_value_head', 'value_head']: # these return tuples
                show(f"SKIPPING {key}:{type(submod)}")
                continue
            if isinstance(submod, torch.nn.ModuleList):
                show(f"INDIVIDUALLY WRAPPING {key}:{type(submod)}")
                for i, subsubmod in enumerate(submod):
                    new_submod = HookedModuleWrapper(subsubmod, name=f'{key}.{i}', recursive=True, top_level=False)
                    submod[i] = new_submod
                continue
            # print(f'wrapping {key}:{type(submod)}')
            new_submod = HookedModuleWrapper(submod, name=key, recursive=True, top_level=False)
            self.mod.__setattr__(key, new_submod)

    def forward(self, *args, **kwargs):
       result = self.mod.forward(*args, **kwargs)
       if not self.hook_self:
           return result
       assert isinstance(result, Tensor)
       return self.hook_point(result)


class HookedKataGoWrapper(HookedModuleWrapper):
    """
    Mirrors this alternate implementation of subnetwork probing, to avoid using MaskedHookPoints:
        https://github.com/rhaps0dy/Automatic-Circuit-Discovery/blob/adria/work-trial/subnetwork_probing/train.py
    """

    mod: KataModel
    mask_logits: torch.nn.ParameterList
    mask_logits_names: List[str]
    _mask_logits_dict: Dict[str, torch.nn.Parameter]

    def __init__(self, model:KataModel, beta=2 / 3, gamma=-0.1, zeta=1.1, mask_init_p=0.9):
        super().__init__(model, name='model', recursive=True, hook_self=False, top_level=True)
        self.setup_masks(beta, gamma, zeta, mask_init_p)

    def setup_masks(self, beta, gamma, zeta, mask_init_p):
        self.mask_logits = torch.nn.ParameterList()
        self.mask_logits_names = []
        self._mask_logits_dict = {}

        self.beta = beta
        self.gamma = gamma
        self.zeta = zeta
        self.mask_init_p = mask_init_p

        # Copied from subnetwork probing code. Similar to log odds (but not the same)
        p = (self.mask_init_p - self.gamma) / (self.zeta - self.gamma)
        mask_init_constant = math.log(p / (1 - p))
        for i, block in enumerate(self.mod.blocks):
            mask_name = f"mod.blocks.{i}.mod.normactconvq.mod.act.hook_point"
            mask_dim = block.mod.normactconvq.mod.conv.mod.in_channels
            self.mask_logits.append(torch.nn.Parameter(torch.zeros((mask_dim,1,1), dtype=torch.float32) + mask_init_constant))
            self.mask_logits_names.append(mask_name)
            self._mask_logits_dict[mask_name] = self.mask_logits[-1]

    def sample_mask(self, mask_name):
        """Samples a binary-ish mask from the mask_scores for the particular `mask_name` activation"""
        mask_scores = self._mask_logits_dict[mask_name]
        uniform_sample = torch.zeros_like(mask_scores).uniform_().clamp_(0.0001, 0.9999)
        s = torch.sigmoid((uniform_sample.log() - (1 - uniform_sample).log() + mask_scores) / self.beta)
        s_bar = s * (self.zeta - self.gamma) + self.gamma
        mask = s_bar.clamp(min=0.0, max=1.0)
        return 1 - 2 * mask  # 0 -> keep, 1 -> invert
    
    def regularization_loss(self) -> torch.Tensor:
        center = self.beta * math.log(-self.gamma / self.zeta)
        per_parameter_loss = [
            torch.sigmoid(scores - center).mean()
            for scores in self.mask_logits
        ]
        return torch.mean(torch.stack(per_parameter_loss))

    def mask_logits_names_filter(self, name):
        return name in self.mask_logits_names
    
    def activation_mask_hook(self, hook_point_out: torch.Tensor, hook: HookPoint):
        mask = self.sample_mask(hook.name)
        # print(f"trying to multiply mask {mask.shape} with hook_point_out {hook_point_out.shape}")
        out = mask * hook_point_out
        return out

    def fwd_hooks(self) -> List[Tuple[str, Callable]]:
        return [(n, self.activation_mask_hook) for n in self.mask_logits_names]

    def with_fwd_hooks(self) -> ContextManager[Self]:
        return self.hooks(self.fwd_hooks())

    def freeze_weights(self):
        for p in self.mod.parameters():
            p.requires_grad = False


# %%

def mask_flippedness(wrapped_model:HookedKataGoWrapper) -> np.ndarray:
    """
    Gives a complexity score to the mask, adjusting the
    number of things flipped for total number of mask items...
    """
    sums = np.zeros(len(wrapped_model.mask_logits))
    for i, mask_logits in enumerate(wrapped_model.mask_logits):
        mask = wrapped_model.sample_mask(wrapped_model.mask_logits_names[i]).detach().cpu().numpy().flatten()
        sums[i] = np.sum((1 - mask) / 2)
    return sums


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

# %%

class HookedKataTrainingObject:
    def __init__(self, config, device, board_size=19):
        self.device = device
        self.board = Board(board_size)
        self.features = Features(config, board_size)
        self.loc_to_move_map = torch.zeros(362, dtype=torch.int64)
        for i in range(self.board.arrsize):
            entry = self.features.loc_to_tensor_pos(i, self.board)
            if 0 <= entry < 362:
                self.loc_to_move_map[entry] = i

    # copied from play
    def get_model_outputs(self, model:HookedKataGoWrapper, bin_input_data, global_input_data, annotated_values=None):
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
            bin_input_data.to(self.device),
            global_input_data.to(self.device),
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
        reshaped_nan_mask = torch.gather(nan_mask, 1, repeat(self.loc_to_move_map, 'loc -> batch loc', batch=batch_size))
        policy0_logits = policy_logits[:, 0, :] # batch, loc
        policy0_logits[reshaped_nan_mask] = -torch.inf # Prevent policy from making illegal moves

        policy0 = torch.nn.functional.softmax(policy0_logits,dim=1) # batch, loc
        # policy_inverted = torch.nn.functional.softmax(- policy_logits[0,:], dim=0)
        value = torch.nn.functional.softmax(value_logits,dim=0).detach().cpu().numpy()

        # Assume that all moves are legal and handle illegal moves in loss_fn
        probs0 = torch.zeros((batch_size, self.board.arrsize))
        for batch_idx in range(batch_size):
            for i in range(policy0.shape[1]):
                move = self.features.tensor_pos_to_loc(i,self.board)
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

    def loss_fn(self, policy_probs:Tensor, annotated_values:Tensor, pla:int):
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

    def get_losses(self, model, batch):
        bin_input_data, global_input_data, pla = batch["bin_input_data"], batch["global_input_data"], batch["pla"]
        policy_data = self.get_model_outputs(model, bin_input_data, global_input_data, batch["annotated_values"])
        losses = self.loss_fn(policy_data["probs0"], batch["annotated_values"], pla)
        return losses
# %%
