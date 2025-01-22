from package_Dec03 import p07_SystemError_vis
import pandas as pd
import json
from itertools import product


def n_movement_after_(n_movement):
    if n_movement == "init":
        return 0
    else:
        return n_movement + 1

def main(config):
    p07 = p07_SystemError_vis.SystemError_vis(is_df_default=False)

    random_state = config["random_state"]
    exp_id = config["exp_id"]
    n_movement = config["n_movement"]
    n_movement_after = n_movement_after_(n_movement=n_movement)

    dir_path = f"output_Jan20/data/{exp_id}"
    p07.df = pd.read_csv(f"{dir_path}/01_X/{random_state}_{n_movement}.csv.gz", index_col=0)
    p07.gmm_params = {
        "centers": pd.read_csv(f"{dir_path}/05_gmm_centers/{random_state}_{n_movement_after}.csv.gz", index_col=0),
        "pred": pd.read_csv(f"{dir_path}/03_cluster/{random_state}_{n_movement_after}.csv.gz", index_col=0)
    }
    p07