import os
import json
from pprint import pprint

import numpy as np

path = "fine_tuning_fc"

folders = os.listdir(path)

val_losses = []
sub_folders = []
for s in folders:
    subfolder_path = os.path.join(path, s, "metrics.json")
    with open(subfolder_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)
    val_losses.append(metrics["val_losses"][-1])
    sub_folders.append(subfolder_path)

print("Total number of hyperparameters: ", len(val_losses))
print(min(val_losses))
min_index = val_losses.index(min(val_losses))
recommended_path = sub_folders[min_index]

with open(recommended_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)

pprint(metrics["neural net"])

pprint(metrics["val_losses"])
