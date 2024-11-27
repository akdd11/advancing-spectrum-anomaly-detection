import hydra
import numpy as np
import os
import pickle as pkl
import sys
from tqdm import trange

# find the location of the repository and add it to the path to import the modules
repo_name = "advancing-spectrum-anomaly-detection"
module_path = __file__[: __file__.find(repo_name) + len(repo_name)]
sys.path.append(os.path.abspath(module_path))

# pl_tuils required to load pathloss map results from pickle file
from src.utils.pl_utils import PathLossMap, PathLossMapCollection
from src.utils.radiomap_utils import RadioMap, Transmitter

# Load config file which contains the parameterss into cfg object
hydra.initialize(version_base=None, config_path="conf")
cfg = hydra.compose(config_name="radio_map_generation")


# Load the pathloss maps --------------------------------------------------------------------------

dataset_nr = cfg.dataset_nr
dataset_dir = os.path.join(module_path, "datasets")

scene_nr = cfg.sionna.scene_nr
pl_filename = f"scene{scene_nr}_PLdataset{dataset_nr}.pkl"  # pathloss dataset (load)
rm_filename = f"scene{scene_nr}_RMdataset{dataset_nr}.pkl"  # radiomap dataset (save)

with open(os.path.join(dataset_dir, pl_filename), "rb") as fin:
    pl_results = pkl.load(fin)

# Configuration -----------------------------------------------------------------------------------

tx_power = cfg.tx_power  # transmit power in dBm
range_num_tx = [
    cfg.min_num_tx,
    cfg.max_num_tx,
]  # minimum and maximum number of transmitters
dist_num_tx = "uniform"  # distribution of the number of transmitters

range_jam_power = [tx_power, tx_power]  # minimum and maximum jamming power in dBm
dist_jam_power = "uniform"  # distribution of the jamming power
range_num_jam = [
    cfg.min_num_jam,
    cfg.max_num_jam,
]  # minimum and maximum number of jammers
dist_num_jam = "uniform"  # distribution of the number of jammers

# Generate the radiomaps --------------------------------------------------------------------------

print(f"Path loss dataset filename: {pl_filename}")

radiomaps = []
for i in trange(cfg.num_radiomaps):
    radio_map = RadioMap(
        pl_results.config["scene_size"][:2],
        pl_results.config["resolution"],
        noise_floor=cfg.sionna.noise_floor,
    )
    # there can only be one transmitter or jammer at a certain position
    pl_map_idxs = list(range(len(pl_results.pathlossmaps)))

    # Choose the number of transmitters and generate the corresponding radio maps
    num_tx = np.random.randint(range_num_tx[0], range_num_tx[1] + 1)
    for i_tx in range(num_tx):
        # Choose a random path loss map
        idx = np.random.choice(pl_map_idxs)
        pl_map_idxs.remove(idx)

        radio_map.add_transmitter(
            "tx",
            pl_results.pathlossmaps[idx].tx_pos,
            tx_power,
            pl_results.pathlossmaps[idx].pathloss,
        )

    # Do the same for the jammers
    jam_power = np.random.uniform(range_jam_power[0], range_jam_power[1])
    num_jam = np.random.randint(range_num_jam[0], range_num_jam[1] + 1)
    for i_jam in range(num_jam):
        # Choose a random path loss map
        idx = np.random.choice(pl_map_idxs)
        pl_map_idxs.remove(idx)

        radio_map.add_transmitter(
            "jammer",
            pl_results.pathlossmaps[idx].tx_pos,
            jam_power,
            pl_results.pathlossmaps[idx].pathloss,
        )

    # Convert very small and -inf values to noise floor
    if cfg.sionna.noise_floor is not None:
        radio_map.apply_noise_floor()

    # convert radio_map to dBm
    radiomaps.append(radio_map)

# Save the radio maps to a pickle file
with open(os.path.join(dataset_dir, rm_filename), "wb") as fout:
    pkl.dump(radiomaps, fout)

# Adding the description of the dataset to a text file
config = pl_results.config
with open(os.path.join(dataset_dir, rm_filename[:-4] + ".txt"), "w") as f_descr:
    for k in config.keys():
        print(f"{k}: {config[k]}", file=f_descr)
