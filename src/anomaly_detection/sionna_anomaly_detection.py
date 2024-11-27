# Module import
import os
import sys

import hydra
from tqdm import tqdm

# Own modules
repo_name = "advancing-spectrum-anomaly-detection"
module_path = __file__[: __file__.find(repo_name) + len(repo_name)]
sys.path.append(os.path.abspath(module_path))

from src.anomaly_detection import anomaly_detection_utils

# Load config file which contains the parameters into cfg object
hydra.initialize(version_base=None, config_path="conf")
cfg = hydra.compose(config_name="sionna_anomaly_detection")


print(f"Method: {cfg.method}")

scene_nr = cfg.scene_nr
dataset_nr = cfg.dataset_nr

# Run the anomaly detection for different noise levels

for noise_std in tqdm(cfg.meas_noise_std):

    anomaly_detection_utils.sionna_anomaly_detection(
        cfg.method,
        cfg.outlier_probability,
        scene_nr,
        dataset_nr,
        cfg,
        noise_std,
        cfg.probability,
        cfg.verbosity,
    )
