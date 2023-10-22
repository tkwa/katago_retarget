
Attempting to intervene on activations to cause KataGo to output the worst move.

## How to replicate

* Download a KataGo network from `https://katagotraining.org/networks/`.
* Run `download_sgf.py` to download KataGo training games.
* Run `annotate_sgf.py` to annotate each game with value head evaluations for all possible moves after each position. This takes about 10 hours for 1000 games on an A10 GPU.
* ...