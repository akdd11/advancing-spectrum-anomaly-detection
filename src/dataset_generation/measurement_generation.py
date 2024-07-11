import os

# GPU config
gpu_num = "0" # Use "" to use the CPU
os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_num}"

from distutils.spawn import find_executable
import hydra
import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl
import seaborn as sns
import shutil
import sionna.rt as srt
import sys
import time
from tqdm import tqdm
import warnings


# find the location of the repository and add it to the path to import the modules
repo_name = 'advancing-spectrum-anomaly-detection'
module_path = __file__[:__file__.find(repo_name)+len(repo_name)]
sys.path.append(os.path.abspath(module_path))

from src.utils.pl_utils import PathLossMap, PathLossMapCollection  # required to load pathloss map results from pickle file
from src.utils.pl_utils import enclosed_in_obstacle, create_sionna_sample, get_obstacle_mask
from src.utils.ml_utils import generate_df, train_model, prepare_input
from src.utils import radiomap_utils

# Load config file which contains the parameterss into cfg object
hydra.initialize(version_base=None, config_path='conf')
cfg = hydra.compose(config_name='measurement_generation')

# Plot configuration
latex_installed = True if find_executable('latex') else False   # check if latex is installed
plt.rcParams['text.usetex'] = latex_installed
plt.rcParams['font.size'] = 15
plt.rcParams['font.family'] = 'serif'

# Define configuration -----------------------------------------------------------------------------

# Validity check for parameters
if cfg.dt_generation not in ['accurate', 'ml']:
    raise ValueError('Invalid value for dt_generation. Only "accurate" and "ml" are allowed.')

measurement_method = cfg.measurement_method     # 'grid' or 'custom1' or 'random'
if measurement_method == 'grid':
    grid_size = cfg.grid_size               # grid size of the measurements in meters

pos_std = cfg.tx_pos_inaccuracy_std    # std of the inaccuracy of the tx position

# Load dataset -------------------------------------------------------------------------------------

rm_dataset_nr = cfg.rm_dataset_nr       # number of the path loss and radio map dataset to load
meas_dataset_nr = cfg.meas_dataset_nr   # number of the measurements dataset to save
dataset_dir = os.path.join(module_path, 'datasets')

scene_nr = cfg.scene_nr

if not cfg.use_original_pl_maps:
    scene_path = os.path.join(os.path.dirname(__file__),'..','..','scenes',f'scene{scene_nr}','scene.xml')

    scene = srt.load_scene(scene_path)

pl_filename = f'scene{scene_nr}_PLdataset{rm_dataset_nr}.pkl'  # pathloss dataset (load)
rm_filename = f'scene{scene_nr}_RMdataset{rm_dataset_nr}.pkl'  # radiomap dataset (load)
measurements_filename = f'scene{scene_nr}_measurements{meas_dataset_nr}.pkl'  # measurements dataset (save)

if cfg.save_radio_maps:
    # filename where the plots are saved
    figure_path = os.path.join(module_path, 'figures', 'radiomaps')
    if not os.path.exists(figure_path):
        os.makedirs(figure_path)

    if cfg.dt_generation == 'ml':
        warnings.warn('No radiomaps created for dt_generation=ml. No plots will be saved.', UserWarning)


# Check if the file already exists
if os.path.isfile(os.path.join(dataset_dir, measurements_filename)):
    print(f'Dataset {measurements_filename} already exists. Overwrite? (y/n)')
    answer = input()
    if answer == 'y':
        os.remove(os.path.join(dataset_dir, measurements_filename))
    else:
        sys.exit()

print(f'Generating {measurements_filename} ...')

with open(os.path.join(dataset_dir, pl_filename), 'rb') as fin:
    plmc = pkl.load(fin)

with open(os.path.join(dataset_dir, rm_filename), 'rb') as fin:
    radiomaps = pkl.load(fin)


# If sionna, extract and apply further configurations from path loss map collection

if not cfg.use_original_pl_maps:
    config = plmc.config
    
    # Configure antenna array for all transmitters and receivers
    scene.tx_array = srt.PlanarArray(num_rows=1,
                                     num_cols=1,
                                     vertical_spacing=0.5,
                                     horizontal_spacing=0.5,
                                     pattern='iso',
                                     polarization='VH')
    scene.rx_array = srt.PlanarArray(num_rows=1,
                                     num_cols=1,
                                     vertical_spacing=0.5,
                                     horizontal_spacing=0.5,
                                     pattern='iso',
                                     polarization='VH')

# Collect errors between digital twin (without jammer) and the radio map measured (eventually containing jammer)

jammed_diffs = {'mean': [], 'median': []}
not_jammed_diffs = {'mean': [], 'median': []}

# Define the measurement points
meas_x, meas_y = radiomap_utils.generate_measurement_points(measurement_method,
                                                            radiomaps[0].radio_map.shape)

obstacle_mask = get_obstacle_mask(scene_nr, scene.size.numpy().astype(int), config['rx_height']) 


# Prepare ML dataset and train model
if cfg.dt_generation == 'ml':
    print(f"Generating dataset for ML : Scene{scene_nr}")
    df = generate_df(meas_x, meas_y, scene_nr, cfg.ml_pl_dataset_nr, measurement_method,
                     use_dist=cfg.ml_use_distances)
    
    print(f"\nModel Training... (Started at: {time.ctime(time.time())})\n")
    start = time.time()
    model, scaler_x, scaler_y = train_model(df, validate=True, use_dist=cfg.ml_use_distances,
                                            num_sus=len(meas_x))
    end = time.time()

    print(f'Time elapsed in model training: {round((end-start)/60,5)} minutes')

# Get resolution of radio maps
resolution = radiomaps[0].resolution
        
# A collection in which the measurements are stored
measurement_collection = radiomap_utils.MeasurementCollection(measurement_method, meas_x, meas_y)

def add_localization_error():
    # Add localization error to the transmitter position with std given in yaml file
    # It is ensured, that the estimated transmitter position is not inside an obstacle
    # or outside of the scene.

    while(True):
        # ensure the estimated transmitter position is not inside an obstacle
        r_err = np.random.normal(0, pos_std)
        phi_err = np.random.uniform(0, 2*np.pi)
        x_err = r_err * np.cos(phi_err)
        y_err = r_err * np.sin(phi_err)
        tx_pos_est = [tx.tx_pos[0]+x_err, tx.tx_pos[1]+y_err, tx.tx_pos[2]]
        
        if enclosed_in_obstacle(tx_pos_est, scene_nr):
            continue
        else:
            break

    # ensure, that the transmitter is not outside the room (otherwise, big error due to wall)
    tx_pos_est[0] = np.clip(tx_pos_est[0], 0.05, plmc.config['scene_size'][0]-0.05)
    tx_pos_est[1] = np.clip(tx_pos_est[1], 0.05, plmc.config['scene_size'][1]-0.05)

    return tx_pos_est

start = time.time()

for rm_idx, rm_orig in enumerate(tqdm(radiomaps)):

    # Create digital twin of the radio environment
    if cfg.use_original_pl_maps:
        rm_dt = radiomap_utils.RadioMap(rm_orig.radio_map.shape, resolution)
        for tx in rm_orig.transmitters:
            pathlossmap = plmc.pathlossmap_for_tx_pos(tx.tx_pos)
            rm_dt.add_transmitter('tx', tx.tx_pos, tx.tx_power, pathlossmap)
    else:

        if cfg.dt_generation == 'accurate':

            # The DT is generated by running ray tracing for each transmitter present in each radio map

            rm_dt = radiomap_utils.RadioMap(rm_orig.radio_map.shape, resolution, rm_orig.noise_floor)
            for tx in rm_orig.transmitters:
                # Recreate path loss map for estimated transmitter position
                tx_pos_est = add_localization_error()
                pathlossmap = create_sionna_sample(config, scene, scene_nr, tx_pos=tx_pos_est).pathloss

                rm_dt.add_transmitter('tx', tx_pos_est, tx.tx_power, pathlossmap)

            # Apply the noise floor for very small values (noise floor value already set at initialization of radio map)
            rm_dt.apply_noise_floor()

            if cfg.plot_radio_maps or cfg.save_radio_maps:
                # mask the pixels that are inside obstacles
                rm_orig_obstacles = np.copy(rm_orig.radio_map)
                rm_orig_obstacles[obstacle_mask] = np.nan
                rm_dt_obstacles = np.copy(rm_dt.radio_map)
                rm_dt_obstacles[obstacle_mask] = np.nan
                # find minimum and maximum value for both radio maps
                v_min = np.nanmin([np.nanmin(rm_orig_obstacles), np.nanmin(rm_dt_obstacles)])
                v_max = np.nanmax([np.nanmax(rm_orig_obstacles), np.nanmax(rm_dt_obstacles)])
                if cfg.save_radio_maps:
                    filename_orig = os.path.join(figure_path, f'radiomap_orig_{rm_idx}.png')
                    filename_dt = os.path.join(figure_path, f'radiomap_dt_{rm_idx}.png')
                    filename_diff = os.path.join(figure_path, f'radiomap_diff_{rm_idx}.png')
                else:
                    filename_orig = None
                    filename_dt = None
                    filename_diff = None

                # plot the radio maps (original and DT)
                rm_orig.show_radio_map(rm_type='orig', vmin=v_min, vmax=v_max,
                                        obstacle_mask=obstacle_mask,
                                        show_plot=cfg.plot_radio_maps,
                                        filename=filename_orig)
                rm_dt.show_radio_map(rm_type='dt', vmin=v_min, vmax=v_max,
                                        obstacle_mask=obstacle_mask,
                                        show_plot=cfg.plot_radio_maps,
                                        filename=filename_dt)

                # plot the difference
                radiomap_utils.plot_radio_map_difference(rm_orig, rm_dt, plot_orig_tx=False,
                                                         plot_dt_tx=False,
                                                         meas_x=meas_x, meas_y=meas_y,
                                                         res_offset=rm_orig.plot_offset,
                                                         obstacle_mask=obstacle_mask,
                                                         show_plot=cfg.plot_radio_maps,
                                                         filename=filename_diff)
            
        elif cfg.dt_generation == 'ml':

            # In this case, not raytracing is performed to generate the digital twin,
            # but a pre-learned model is used to estimate the RSS at the SU locations.

            measurements_dt = np.zeros(len(meas_x)).reshape(1,-1) # RSS, for the beginning in linear scale
           
            for tx in rm_orig.transmitters:
                tx_pos_est = add_localization_error()
                
                tx_input = prepare_input(tx_pos_est, meas_x, meas_y, scaler_x,
                                         measurement_method=measurement_method,
                                         use_dist=cfg.ml_use_distances)
                pred_loss = model.predict(tx_input)
                pred_loss = scaler_y.inverse_transform(pred_loss)
                rx_power_lin =  np.power(10, (tx.tx_power - pred_loss)/10)
                
                measurements_dt += rx_power_lin 
                    
            measurements_dt = 10 * np.log10(measurements_dt)     # convert to dBm
    
    measurements_orig = radiomap_utils.do_measurements(rm_orig, meas_x, meas_y)
    if cfg.dt_generation == 'accurate':
        measurements_dt = radiomap_utils.do_measurements(rm_dt, meas_x, meas_y)

    measurement_collection.add_measurement(rm_orig.transmitters, rm_orig.jammers, measurements_orig, measurements_dt)

    if len(rm_orig.jammers) == 1:
        jammed_diffs['mean'].append(np.mean(measurements_orig - measurements_dt))
    else:
        not_jammed_diffs['mean'].append(np.mean(measurements_orig - measurements_dt))
        
end = time.time()
print(f'Time elapsed in DT generation: {round((end-start)/60,5)} minutes')

with open(os.path.join(dataset_dir, measurements_filename), 'wb') as f_out:
    pkl.dump(measurement_collection, file=f_out)

# save description file of measurements
if os.path.isfile(os.path.join(dataset_dir, measurements_filename[:-4]+'.txt')):
    # if file already exists, shutil.copyfile raises exception -> delete first
    os.remove(os.path.join(dataset_dir, measurements_filename[:-4]+'.txt'))
shutil.copyfile(os.path.join(dataset_dir, rm_filename[:-4]+'.txt'),
                os.path.join(dataset_dir, measurements_filename[:-4]+'.txt'))
with open(os.path.join(dataset_dir, measurements_filename[:-4]+'.txt'), 'a') as f_out:
    print(f'\nmeasurement_method: {measurement_method}', file=f_out)
    if measurement_method == 'grid':
        print(f'grid_size: {grid_size}', file=f_out)    
    print(f'tx_pos_inaccuracy_std: {pos_std}', file=f_out)
    print(f'dt_generation: {cfg.dt_generation}', file=f_out)

# Create a density plot of the difference between the original and the digital twin measurements
sns.kdeplot(jammed_diffs['mean'], label='Jammed')
sns.kdeplot(not_jammed_diffs['mean'], label='Not jammed')
if latex_installed:
    plt.xlabel(r'$\overline{\Delta}$ $[\mathrm{dB}]$')
else:
    plt.xlabel('Mean difference [dB]')
plt.grid()
plt.legend()
plt.tight_layout()
if cfg.save_density_plot:
    filename = os.path.join(module_path, 'figures',
                            f'error_density_{cfg.dt_generation}_su{config["rx_height"]}m.png')
    plt.savefig(filename, dpi=300)
plt.show()
