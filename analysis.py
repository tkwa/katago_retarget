# %%
from snp_utils import HookedKataGoWrapper
from KataGo.python.load_model import load_model
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

def visualize_mask(wrapped_model:HookedKataGoWrapper):
    # Make histograms of mask values
    fig, axs = plt.subplots(4, 5)
    for i, ax in enumerate(axs.flat[:len(wrapped_model.mask_logits)]):
        mask = wrapped_model.sample_mask(wrapped_model.mask_logits_names[i]).detach().cpu().numpy().flatten()
        ax.hist(mask, bins=20)
        ax.set_title(f"Layer {i}")
    plt.show()
    print(f"Mean mask value: {torch.tensor([wrapped_model.sample_mask(n).mean() for n in wrapped_model.mask_logits_names]).mean()}")

visualize_mask(wrapped_model)
# %%