# %%
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader
import time
import wandb
import argparse
import gc
# import cProfile
# import pstats
# from torch.profiler import profile, record_function, ProfilerActivity

from snp_utils import HookedKataGoWrapper, HookedKataTrainingObject, KataPessimizeDataset, mask_flippedness
sys.path.append("./KataGo/python")

# from KataGo.python.play import get_outputs
from KataGo.python.load_model import load_model
from tqdm import tqdm


# %%

DATASET_DIR = 'dataset'
CHECKPOINT_FILE = 'kg_checkpoint/kata1-b18c384nbt-s7709731328-d3715293823/model.ckpt'
DEVICE = 'cuda'
pos_len = 19

parser = argparse.ArgumentParser(
    description="Use modified subnetwork probing to train the KataGo policy net to make the worst move")
add = parser.add_argument

add("--dataset_dir", default=DATASET_DIR)
add("--checkpoint_file", default=CHECKPOINT_FILE)
add("--device", default=DEVICE)
add("--no_wandb", action="store_true")
add("--n_epochs", default=100, type=int)
add("--lr", default=0.01, type=float)
add("--regularization_lambda", default=1, type=float)
add("--batch_size", default=96, type=int)

args, unknown = parser.parse_known_args()

# %%

kata_model, swa_model, _ = load_model(args.checkpoint_file, None, device=args.device, pos_len=pos_len, verbose=True)
kata_model.eval()
model_config = kata_model.config

# %%

dataset = KataPessimizeDataset(args.dataset_dir, n_games=800)
val_frac = 0.1
train_set, val_set = torch.utils.data.random_split(dataset, [int(len(dataset)*(1-val_frac)), int(len(dataset)*val_frac)])

# %%
wrapped_model = HookedKataGoWrapper(kata_model).to(args.device)
wrapped_model.freeze_weights()

# %%
def train(wrapped_model:HookedKataGoWrapper, train_loader:DataLoader, val_loader: DataLoader, n_epochs=1, regularization_lambda=1, lr=0.005, use_wandb=True):
    print(f"Starting training for {n_epochs} epochs")
    if use_wandb:
        wandb.init(project="kata-pessimize")
        wandb.watch(wrapped_model)
    print(f"args: {args}")
    training_object = HookedKataTrainingObject(model_config, args.device)
    optimizer = torch.optim.Adam(wrapped_model.parameters(), lr=lr) # TODO change lr
    with wrapped_model.with_fwd_hooks() as hooked_model:
        for epoch in range(n_epochs):
            regrets = []
            regularization_losses = []
            total_losses = []
            hooked_model.train()
            print(f"Starting epoch {epoch}/{n_epochs}")
            for batch in tqdm(train_loader):
                optimizer.zero_grad()
                losses = training_object.get_losses(hooked_model, batch)
                regularization_loss = hooked_model.regularization_loss()
                avg_loss = losses.mean()
                # if epoch==0: print(avg_loss.item())
                total_loss = regularization_lambda * regularization_loss + avg_loss
                total_loss.backward()
                optimizer.step()
                regrets.append(avg_loss.item())
                regularization_losses.append(regularization_loss.item())
                total_losses.append(total_loss.item())
            print(f"Average regret: {np.mean(regrets)}")
            print(f"Average regularization loss: {np.mean(regularization_losses)}")
            mean_mask_value = torch.tensor([wrapped_model.sample_mask(n).mean() for n in wrapped_model.mask_logits_names]).mean()
            if use_wandb:
                val_regrets = []
                hooked_model.eval()
                for batch in tqdm(val_loader):
                    losses = training_object.get_losses(hooked_model, batch)
                    val_regrets.append(losses.mean().item())
                print(f"Average validation regret: {np.mean(val_regrets)}")
                wandb.log({"regret": np.mean(regrets), "val_regret": np.mean(val_regrets), "regularization_loss": np.mean(regularization_losses),
                    "total_loss": np.mean(total_losses), "mean_mask_value": mean_mask_value, "flippedness": mask_flippedness(wrapped_model).sum()})
            if (epoch + 1) % 5 == 0:
                time_str = time.strftime("%Y%m%d-%H%M%S")
                state_dict = wrapped_model.state_dict()
                state_dict["config"] = model_config
                torch.save(state_dict, f"models/model_{epoch}_{time_str}.pth")

    if use_wandb:
        wandb.finish()

# %%

# %%
train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
print(f"{'not using' if args.no_wandb else 'using'} wandb")
# %%
gc.collect()
torch.cuda.empty_cache()
train(wrapped_model, train_loader, val_loader, n_epochs=int(args.n_epochs), lr=args.lr, regularization_lambda=args.regularization_lambda, use_wandb=not args.no_wandb)
# cProfile.run("train(wrapped_model, data_loader)", "output.pstats")
