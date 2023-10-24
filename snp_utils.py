# %%
from copy import deepcopy
import sys
import math
import torch
from torch import Tensor
from typing import List, Dict, Tuple, Callable, ContextManager
from transformer_lens.hook_points import HookPoint, HookedRootModule

sys.path.append("/home/ubuntu/katago_pessimize/KataGo/python")
from KataGo.python.model_pytorch import Model as KataModel
from KataGo.python.load_model import load_model

# Import the subnetwork probing module from Automatic-Circuit-Discovery/acdc
sys.path.append('./Automatic-Circuit-Discovery')
from subnetwork_probing.train import *
# sys.path.append('./Automatic-Circuit-Discovery/subnetwork_probing/transformer_lens/transformer_lens')
# from subnetwork_probing.train import *
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
        self.mod = deepcopy(mod)
        self.hook_self = hook_self
        if hook_self:
            hook_point = HookPoint()
            hook_point.name = name
            self.hook_point = hook_point
        if recursive: self.wrap_hookpoints_recursively()
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
        self.setup_masks()

    def setup_masks(self, beta=2 / 3, gamma=-0.1, zeta=1.1, mask_init_p=0.9):
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
            mask_name = f"blocks.{i}.mod.normactconvq.mod.act.hook_point"
            mask_dim = block.mod.normactconvq.mod.conv.mod.in_channels
            self.mask_logits.append(torch.nn.Parameter(torch.zeros(mask_dim, dtype=torch.float32) + mask_init_constant))
            self.mask_logits_names.append(mask_name)
            self._mask_logits_dict[mask_name] = self.mask_logits[-1]

    def sample_mask(self, mask_name):
        """Samples a binary-ish mask from the mask_scores for the particular `mask_name` activation"""
        mask_scores = self._mask_logits_dict[mask_name]
        uniform_sample = torch.zeros_like(mask_scores).uniform_().clamp_(0.0001, 0.9999)
        s = torch.sigmoid((uniform_sample.log() - (1 - uniform_sample).log() + mask_scores) / self.beta)
        s_bar = s * (self.zeta - self.gamma) + self.gamma
        mask = s_bar.clamp(min=0.0, max=1.0)
        return mask
    
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
        out = mask * hook_point_out + (1 - mask) * self.cache[hook.name]
        return out

    def fwd_hooks(self) -> List[Tuple[str, Callable]]:
        return [(n, self.activation_mask_hook) for n in self.mask_logits_names]

    def with_fwd_hooks(self) -> ContextManager[HookedRootModule]:
        return self.hooks(self.fwd_hooks())

    def freeze_weights(self):
        for p in self.model.parameters():
            p.requires_grad = False
# %%