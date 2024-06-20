''' Contains some utilities to work with path loss maps.'''

import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import seaborn as sns
import sionna.rt as srt
import tensorflow as tf
import yaml 

repo_name = 'advancing-spectrum-anomaly-detection'
module_path = __file__[:__file__.find(repo_name)+len(repo_name)]
sys.path.append(os.path.abspath(module_path))


# Color constants
COLOR_TX_ORIG = 'green'


class PathLossMap:
    '''Class to save a path loss map.
    
    Contains the position of the transmitter and the path loss map.'''


    def __init__(self, tx_pos, pathloss):
        '''Initializes the PathLossMap object.

        tx_pos : tuple
            Position of the transmitter in meters.
        pathloss : numpy.ndarray
            Path loss map in dB.
        '''
        self.tx_pos = tx_pos
        self.pathloss = pathloss

    def show_pathloss_map(self, show_tx_pos=True):
        """Shows the path loss map.
        
        show_tx_pos : bool
            If True, the position of the transmitter is shown in the plot.
        """

        vmax = np.max(self.pathloss[np.isfinite(self.pathloss)]) # set biggest value that does not equal inf as vmax

        sns.heatmap(self.pathloss.T, square=True, vmax=vmax, cbar=True, cbar_kws={'label': 'Path loss [dB]'}) 
        if show_tx_pos:
            plt.plot(self.tx_pos[0], self.tx_pos[1], 'v', color=COLOR_TX_ORIG, markersize=10, label='Transmitter')
        plt.xlabel('x [m]')
        plt.ylabel('y [m]')
        plt.legend()
        plt.gca().invert_yaxis()
        plt.show()


class PathLossMapCollection:
    '''Class to save the results of the path loss map generation.

    Contains the configuration and an array of PathLossMap objects.'''
    def __init__(self, config):
        self.config = config
        self.pathlossmaps = []

    def pathlossmap_for_tx_pos(self, tx_pos):
        '''Returns the path loss map for a given transmitter position.
        
        tx_pos : tuple
            Position of the transmitter in meters.
        '''
        for plm in self.pathlossmaps:
            if np.array_equal(plm.tx_pos, tx_pos):
                return plm.pathloss

        raise ValueError('No path loss map found for the given transmitter position.')



def smooth_path_loss(path_loss, n=5):
    '''Sort out n biggest values and replace them with the n-th biggest value of the path loss map
    and returns the smoothed path loss map.

    In some earlier sionna version, extremely small values for the path loss in proximity to the 
    transmitter were generated. This function serves to cope with this problem.
    
    path_loss : numpy.ndarray
        Path loss map in dB.
    n : int
        Number of values to be sorted out.
    '''

    pl_threshold = np.sort(path_loss.flatten())[-n]
    path_loss[path_loss >= pl_threshold] = pl_threshold
    return path_loss


def create_sionna_sample(config, scene, scene_nr, tx_pos=None, num_rays=int(1e6)):
    """Returns one path loss map object using Sionna.
    
    config : dict
        Configuration parameters.
    scene : sionna.rt.Scene
        Scene object.
    scene_nr : int
        index of the current scene.
    tx_pos : tuple
        "2D Position of the transmitter in meters.
        If None, the transmitter position is randomly chosen.
    num_rays : int
        Nubmer of rays to be launched to compute the coverage map.
    """
    
    scene_size = scene.size.numpy()

    if tx_pos == None:
         while(True):
             tx_pos = [np.random.uniform(0, scene_size[0]),
                       np.random.uniform(0, scene_size[1]),
                       config['tx_height']]
             if enclosed_in_obstacle(tx_pos, scene_nr):
                 continue
             else:
                 break
    elif enclosed_in_obstacle(tx_pos, scene_nr):
        raise ValueError('Given transmitter position is enclosed within an obstacle.')
            
    tx = srt.Transmitter(name=f'tx', position=tx_pos)
    scene.add(tx)

    # create coverage map
    cm = scene.coverage_map(max_depth=config['cm_max_depth'],
                            cm_center=tf.cast([scene_size[0]/2, scene_size[1]/2, config['rx_height']], dtype=tf.float32),
                            cm_orientation=tf.cast([0,0,0], dtype=tf.float32),
                            cm_size=scene.size[:2],                                       # CM only 2D, whereas scene has a 3D size
                            cm_cell_size=(config['cell_size'], config['cell_size']),      # Grid size of coverage map cells in m
                            diffraction=config['diffraction'],
                            edge_diffraction=config['edge_diffraction'],
                            num_samples=config['num_samples'])
    
    scene.remove(tx.name)

    path_loss_db = -10 * np.log10(cm.as_tensor().numpy()[0].T) # minus to convert gain to loss, transpose as otherwise x and y are swapped
    plm = PathLossMap(tx_pos, path_loss_db)
    
    return plm


def enclosed_in_obstacle(tx_pos, scene_nr):
    """Checks if a point (e.g., a transmitter) is enclosed within any obstacle.

    tx_pos : list (int)
        List containing x, y and z coordinates of the transmitter.
    scene_nr : int
        Index of the current scene.
    """

    # Only read the scene description if it has not been read before
    if not hasattr(enclosed_in_obstacle, '_obstacles'):
        path = os.path.join(os.path.dirname(__file__),'..','..','scenes',f'scene{scene_nr}','scene_attributes.yaml')
    
        # Load the YAML file
        with open(path, 'r') as file:
            scene_description = yaml.safe_load(file)
        obstacles = scene_description.get('obstacles', {})

        # Attach the obstacles to the function
        enclosed_in_obstacle._obstacles = obstacles

    if enclosed_in_obstacle._obstacles == {}:     # No obstacles
        return False

    for obstacle in enclosed_in_obstacle._obstacles.values():
        x, y, z = obstacle['anchor_point']
        w, l, h = obstacle['edge_length']
        if x<= tx_pos[0] <= (x+w) and y<= tx_pos[1] <= (y+l) and tx_pos[2] <= h:
            return True
        else:
            continue
    return False


def get_obstacle_mask(scene_nr, scene_size, height):
    """
    Returns a boolean mask of the scene where obstacles are True
    and free space is False.

    Input
    ------
    scene_nr : int
        Index of the current scene.
    scene_size : tuple
        Size of the scene in meters.
    height : float
        Height to check for obstacles.

    Output
    ------
    mask : np.ndarray
        Boolean mask of the scene.
    """

    mask = np.zeros([scene_size[0], scene_size[1]], dtype=bool)

    for i in range(scene_size[0]):
        for j in range(scene_size[1]):
            if enclosed_in_obstacle([i, j, height], scene_nr):
                mask[i, j] = True

    return mask

