# %%

# %%
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import numpy as np

import sys
sys.path.append("..")
sys.path.append("../KataGo/python")
print(sys.path)
from snp_utils import HookedKataGoWrapper, KataPessimizeDataset, HookedKataTrainingObject, mask_flippedness

from KataGo.python.load_model import load_model
from KataGo.python.model_pytorch import Model

# %%

"""
Test the hooked model's value head. What's the correlation between the hooked value head output
and the original output?
"""
DATASET_DIR = '../dataset'
HOOKED_MODEL_FILE = '../models/model_34_20231029-094834.pth'
CHECKPOINT_FILE = '../kg_checkpoint/kata1-b18c384nbt-s7709731328-d3715293823/model.ckpt'
DEVICE = 'cuda'
pos_len = 19

kata_model, swa_model, _ = load_model(CHECKPOINT_FILE, None, device=DEVICE, pos_len=pos_len, verbose=True)


# %%
state_dict = torch.load(HOOKED_MODEL_FILE,map_location="cpu")
model_config = torch.load(CHECKPOINT_FILE, map_location="cpu")["config"]
inner_model = Model(model_config,pos_len)
inner_model.initialize()
wrapped_model = HookedKataGoWrapper(inner_model).to(DEVICE)
wrapped_model.load_state_dict(state_dict)
# %%

dataset = KataPessimizeDataset(DATASET_DIR, n_games=800)
val_frac = 0.1
_, val_set = torch.utils.data.random_split(dataset, [int(len(dataset)*(1-val_frac)), int(len(dataset)*val_frac)])

# %%

training_object = HookedKataTrainingObject(model_config, DEVICE)

# %%
loader = DataLoader(val_set, batch_size=6, shuffle=True, num_workers=1)
def compare_model_values(model1, model2, batch) -> np.ndarray:
    with torch.no_grad():
        model1.eval()
        model2.eval()
        outputs1 = []
        outputs2 = []
        i = 0
        for batch in tqdm(loader):
            i += 1
            bin_input_data, global_input_data, pla = batch["bin_input_data"], batch["global_input_data"], batch["pla"]
            model1_value = training_object.get_model_outputs(model1, bin_input_data, global_input_data, batch['annotated_values'])['value']
            model2_value = training_object.get_model_outputs(model2, bin_input_data, global_input_data, batch['annotated_values'])['value']
            outputs1.append(model1_value[:, 0])
            outputs2.append(model2_value[:, 0])
            if i >= 100: break
        outputs1 = np.concatenate(outputs1)
        outputs2 = np.concatenate(outputs2)

    return outputs1, outputs2
# %%
def scatter(outputs1, outputs2):
    plt.scatter(outputs1, outputs2)
    plt.xlabel("Original value head output")
    plt.ylabel("Hooked value head output")
    plt.title("Original vs hooked value head output")
    plt.show()
    stacked = np.vstack([outputs1, outputs2])
    # print(f"{stacked.shape=}")
    correlation = torch.corrcoef(torch.tensor(stacked))[0, 1]
    print(f"Correlation: {correlation}")


# %%

with wrapped_model.with_fwd_hooks() as hooked_model:
    outputs = compare_model_values(kata_model, hooked_model, val_set)
# %%
scatter(*outputs)
# %%