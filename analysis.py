# %%
from snp_utils import HookedKataGoWrapper, mask_flippedness
import numpy as np
import torch
import matplotlib.pyplot as plt
from KataGo.python.model_pytorch import Model


# %%
CHECKPOINT_FILE = 'models/model_34_20231029-094834.pth'
CONFIG_MODEL_FILE = 'kg_checkpoint/kata1-b18c384nbt-s7709731328-d3715293823/model.ckpt'
DEVICE = 'cuda'
pos_len = 19


# %%
# kata_model, swa_model, _ = load_model(CHECKPOINT_FILE, None, device=DEVICE, pos_len=pos_len, verbose=True)


state_dict = torch.load(CHECKPOINT_FILE,map_location="cpu")
model_config = torch.load(CONFIG_MODEL_FILE, map_location="cpu")["config"]
kata_model = Model(model_config,pos_len)
kata_model.initialize()
wrapped_model = HookedKataGoWrapper(kata_model).to(DEVICE)
wrapped_model.load_state_dict(state_dict)

# %%
mask_flippedness(wrapped_model)

# %%

def graph_mask_complexity(wrapped_model:HookedKataGoWrapper):
    # Make a bar graph of mask complexity
    sums = mask_flippedness(wrapped_model)
    plt.bar(np.arange(len(sums)), sums)
    short_names = ['.'.join(n.split('.')[1:3]) for n in wrapped_model.mask_logits_names]
    plt.xticks(np.arange(len(sums)), short_names, rotation=90)
    plt.title("Total parameters flipped by mask layer")

graph_mask_complexity(wrapped_model)

# %%

def visualize_mask(wrapped_model:HookedKataGoWrapper):
    # Make histograms of mask values
    fig, axs = plt.subplots(6, 3)
    masks = []
    ax: plt.Axes
    for i, ax in enumerate(axs.flat[:len(wrapped_model.mask_logits)]):
        mask = wrapped_model.sample_mask(wrapped_model.mask_logits_names[i]).detach().cpu().numpy().flatten()
        masks.append(mask)
        ax.hist(mask, bins=20)
        # remove y ticks
        ax.set_yticks([])
        ax.set_title(f"Layer {i}", y=0.5)
        # make title inside plot and transparent
    plt.suptitle("Mask values by layer")
    plt.show()
    # Now make one big histogram
    plt.hist(np.concatenate(masks), bins=40)
    plt.title("Mask values, all layers")
    plt.show()
    print(f"Mean mask value: {torch.tensor([wrapped_model.sample_mask(n).mean() for n in wrapped_model.mask_logits_names]).mean()}")

visualize_mask(wrapped_model)
# %%