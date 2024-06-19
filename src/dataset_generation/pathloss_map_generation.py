'''For a given scene, this script generates a dataset of path loss maps for a given number of random transmitter positions.'''

import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1"

import hydra
import pickle as pkl
import sys
import tensorflow as tf
from tqdm import trange

import sionna.rt as srt

# find the location of the repository and add it to the path to import the modules
repo_name = 'advancing-spectrum-anomaly-detection'
module_path = __file__[:__file__.find(repo_name)+len(repo_name)]
sys.path.append(os.path.abspath(module_path))

from src.utils.pl_utils import PathLossMap, PathLossMapCollection, create_sionna_sample
tf.random.set_seed(42)

# Load config file which contains the parameters into cfg object
hydra.initialize(version_base=None, config_path='conf')
cfg = hydra.compose(config_name='pathloss_map_generation')

# User defined parameters -------------------------------------------------------------------------

f_c = cfg.f_c       # carrier frequency in Hz


scene_nr = cfg.sionna.scene_nr

config = {'scene_nr': scene_nr,
            'f_c': f_c,            # in Hz
            'tx_height': cfg.sionna.tx_height,            # in meters
            'rx_height': cfg.sionna.rx_height,            # in meters
            'cell_size': cfg.sionna.cell_size,            # in meters
            'cm_max_depth': cfg.sionna.cm_max_depth,      # max number of reflections
            'diffraction': cfg.sionna.diffraction,        # diffraction enabled
            'edge_diffraction': cfg.sionna.edge_diffraction,  # edge diffraction enabled
            'num_samples': cfg.sionna.num_samples,        # number of samples launched per path loss map
}
config['resolution'] = config['cell_size']

# GPU configuration -------------------------------------------------------------------------------


physical_devices = tf.config.list_physical_devices('GPU')
try:
  # Disable first GPU
  tf.config.set_visible_devices(physical_devices[1], 'GPU')  
  logical_devices = tf.config.list_logical_devices('GPU')
  print('GPU is used and limited to one device.')
except:
  # Invalid device or cannot modify virtual devices once initialized.
  print('CPU is used.')


# Define paths ------------------------------------------------------------------------------------

# os.path.dirname(__file__) returns the path to the directory where the script is located
dataset_path = os.path.join(os.path.dirname(__file__),'..','..','datasets',f'scene{scene_nr}_PLdataset{cfg.dataset_nr}.pkl')
scene_path = os.path.join(os.path.dirname(__file__),'..','..','scenes',f'scene{scene_nr}','scene.xml')

scene = srt.load_scene(scene_path)
config['scene_size'] = scene.size.numpy()


# Check if the file exists already
if os.path.exists(dataset_path):
    print(f'Dataset {dataset_path} already exists. Overwrite? (y/n)')
    answer = input()
    if answer == 'y':
        os.remove(dataset_path)
    else:
        sys.exit()

# Initialization ----------------------------------------------------------------------------------

scene.tx_array = srt.PlanarArray(num_rows=1,
                                    num_cols=1,
                                    vertical_spacing=0.5,
                                    horizontal_spacing=0.5,
                                    pattern='iso',
                                    polarization='VH')

# Configure antenna array for all receivers
scene.rx_array = srt.PlanarArray(num_rows=1,
                                    num_cols=1,
                                    vertical_spacing=0.5,
                                    horizontal_spacing=0.5,
                                    pattern='iso',
                                    polarization='VH')

scene_size = scene.size.numpy()

plmc = PathLossMapCollection(config)

# Do the actual simulation loop ------------------------------------------------------------------

for idx in trange(cfg.nr_samples):
    plm = create_sionna_sample(config, scene, scene_nr)
    plmc.pathlossmaps.append(plm)


# Save the results -------------------------------------------------------------------------------

with open(dataset_path, 'wb') as fout:
    pkl.dump(plmc, fout)

# Adding the description of the dataset to a text file    
with open(dataset_path[:-4]+'.txt', 'w') as f_descr:
    for k in config.keys():
        print(f'{k}: {config[k]}', file=f_descr)