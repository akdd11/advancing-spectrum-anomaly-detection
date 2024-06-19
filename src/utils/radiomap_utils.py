from distutils.spawn import find_executable
import math
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import warnings

# Color constants
COLOR_TX_ORIG = 'green'
COLOR_TX_DT_FACE = '#C5E0B4' # light green
COLOR_TX_DT_EDGE = 'green'
COLOR_JAMMER_FACE = '#FFF243' # yellow
COLOR_JAMMER_EDGE = '#C84040'# red


latex_installed = True if find_executable('latex') else False   # check if latex is installed


class MeasurementCollection:
    '''Class to save a collection of measurements.

    For each sample in the measurement collection, the following information is saved:
    - The differences between the measurements of the original radio map and the digital twin as a list
    - The regular transmitters in the original radio map as a list
    - The jammers in the original radio map as a list
    
    method : str
        Method to generate the measurement points. Can be 'grid', 'custom' or 'random'.
    meas_x : list
        List of x coordinates of the measurement points.
    meas_y : list
        List of y coordinates of the measurement points.
    '''

    def __init__(self, method, meas_x, meas_y, **kwargs):
        self.method = method
        self.meas_x = meas_x
        self.meas_y = meas_y

        if method == 'grid':
            self.grid_size = kwargs['grid_size']
        
        # list, in which each element is a list of differences between 
        # the measurements of the original radio map and the digital twin
        self.measurements_diff_list = []

        # list, in which each element is a list of the regular transmitters in the original radio map
        self.transmitters_list = []
        # list, in which each element is a list of the jammers in the original radio map
        self.jammers_list = []
        

    def add_measurement(self, transmitters, jammers, measurements_orig, measurements_dt):
        '''Adds a measurement to the collection.

        jammers : list
            List of jammers in the original radio map.
        measurements_orig : numpy.ndarray
            Measurements of the original radio map.
        measurements_dt : numpy.ndarray
            Measurements of the digital twin.
        '''
        self.measurements_diff_list.append(measurements_orig - measurements_dt)
        self.jammers_list.append(jammers)
        self.transmitters_list.append(transmitters)


class RadioMap:
    '''Class to save a radio map.

    transmitters : list
        List of Transmitter objects.
    radio_map : numpy.ndarray
        Radio map in dBm.
    '''

    def __init__(self, shape, resolution, noise_floor=None):
        """Initializes the RadioMap object.
        
        shape : tuple
            Shape of the radio map in meters.
        resolution : float
            Resolution of the radio map in meters.
        noise_floor : float or None
            Noise floor in dBm. If noise_floor
        """
        self.transmitters = []
        self.jammers = []
        self.radio_map = np.zeros(np.array(shape).astype(int))       # in dBm
        self.resolution = resolution

        self.noise_floor = noise_floor


    def apply_noise_floor(self):
        """Applies the noise floor to the radio map.
        
        Each value that is smaller than the noise floor will be set to the noise floor.
        """

        if self.noise_floor is None:
            warnings.warn('Noise floor is applied, but not set.')
            return
        self.radio_map[np.isneginf(self.radio_map)] = self.noise_floor
        self.radio_map[self.radio_map < self.noise_floor] = self.noise_floor


    def add_transmitter(self, tx_type, tx_pos, tx_power, pathloss_map):
        '''Adds a transmitter to the radio map.

        tx_type : str
            Type of the transmitter. Can be 'tx' or 'jammer'.
        tx_pos : tuple
            Position of the transmitter in meters.
        tx_power : float
            Transmit power of the transmitter in dBm.
        pathloss_map : numpy.ndarray
            Pathloss map of the transmitter in dB.
        '''
        if tx_type == 'tx':
            self.transmitters.append(Transmitter(tx_type, tx_pos, tx_power))
        elif tx_type == 'jammer':
            self.jammers.append(Transmitter(tx_type, tx_pos, tx_power))
        else:
            raise ValueError('Type has to be either tx or jammer.')
        
        # handling NaN values
        pathloss_map[np.isnan(pathloss_map)] = float('inf')
        
        if np.any(pathloss_map < 0):
            raise ValueError('Pathloss map contains the loss, i.e., values must be > 0.')
        single_radio_map = tx_power - pathloss_map      # note: pathloss_map values shall be > 0
               
        if (len(self.transmitters) + len(self.jammers)) == 1:
            # when first transmitter is added, radio map is initialized with single_radio_map
            self.radio_map = single_radio_map
        else:
            # add in linear scale and convert back to dBm
            self.radio_map = 10*np.log10(10**(self.radio_map/10) + 10**(single_radio_map/10))


    def show_radio_map(self, plot_transmitters=True, rm_type='orig', **kwargs):
        """Plot the radio map as a heatmap.
        
        plot_transmitters : bool
            If True, transmitters and jammers are plotted on top of the radio map.
        rm_type : str
            Type of the radio map. Can be 'orig' or 'dt'. Default is 'orig'.
            This only influences the labeling and the color of the transmitters.
        (optional) meas_x : list
            List of x coordinates of the measurement points.
        (optional) meas_y : list
            List of y coordinates of the measurement points.
        (optional) vmin : float
            Minimum value of the colorbar.
        (optional) vmax : float
            Maximum value of the colorbar.
        (optional) obstacle_mask : numpy.ndarray
            Mask for obstacles in the radio map. If provided, obstacle values are
            set to NaN.
        """

        if rm_type not in ['orig', 'dt']:
            raise ValueError('rm_type has to be either orig or dt.')

        plt.rcParams['text.usetex'] = latex_installed
        plt.rcParams['font.family'] = 'serif'

        vmin = kwargs['vmin'] if 'vmin' in kwargs else None
        vmax = kwargs['vmax'] if 'vmax' in kwargs else None

        if rm_type == 'orig':
            cbar_label = '$P_\mathrm{rx} [\mathrm{dBm}]$' if latex_installed else 'P_rx [dBm]'
        elif rm_type == 'dt':
            cbar_label = r'$\hat{P}_\mathrm{rx} [\mathrm{dBm}]$' if latex_installed else 'P^hat_rx [dBm]'
        
        radio_map = np.copy(self.radio_map)
        if 'obstacle_mask' in kwargs:
            radio_map[kwargs['obstacle_mask']] = np.nan
        sns.heatmap(radio_map.T, square=True, vmin=vmin, vmax=vmax, cmap='viridis',
                    cbar=True, cbar_kws={'label': cbar_label})        

        if plot_transmitters:

            for transmitter in self.transmitters:
                if rm_type == 'orig':
                    plt.plot(transmitter.tx_pos[0],
                             transmitter.tx_pos[1],
                             'v', color=COLOR_TX_ORIG, markersize=10)
                elif rm_type == 'dt':
                    print(f'Tx pos in dt map: {transmitter.tx_pos}')
                    plt.plot(transmitter.tx_pos[0],
                             transmitter.tx_pos[1],
                             'v', markerfacecolor=COLOR_TX_DT_FACE,
                             markeredgecolor=COLOR_TX_DT_EDGE, markersize=10)
            for jammer in self.jammers:
                print(f'Jammer pos in orig map: {jammer.tx_pos}')
                plt.plot(jammer.tx_pos[0],
                         jammer.tx_pos[1],
                         'X', markerfacecolor=COLOR_JAMMER_FACE,
                         markeredgecolor=COLOR_JAMMER_EDGE, markersize=10)


        if 'meas_x' in kwargs and 'meas_y' not in kwargs:
            raise ValueError('Both meas_x and meas_y have to be given.')
        elif 'meas_x' not in kwargs and 'meas_y' in kwargs:
            raise ValueError('Both meas_x and meas_y have to be given.')
        elif 'meas_x' in kwargs and 'meas_y' in kwargs:
            plt.plot(kwargs['meas_x'], kwargs['meas_y'], 'k.', markersize=3)

        plt.gca().invert_yaxis()

        # Generate legend entries
        if rm_type == 'orig':
            plt.plot([], [], 'v', color=COLOR_TX_ORIG, markersize=10, label='Transmitter')
        else:
            plt.plot([], [], 'v', markerfacecolor=COLOR_TX_DT_FACE,
                      markeredgecolor=COLOR_TX_DT_EDGE,
                      markersize=10, label='DT transmitter')
        if len(self.jammers) > 0:
            plt.plot([], [], 'X', markerfacecolor=COLOR_JAMMER_FACE,
                      markeredgecolor=COLOR_JAMMER_EDGE,
                      markersize=10, linewidth=20, label='Jammer')
        if 'meas_x' in kwargs and 'meas_y' in kwargs:
            plt.plot([], [], 'k.', markersize=3, label='Measurement')
        plt.xlabel('x [m]')
        plt.ylabel('y [m]')
        plt.xticks(np.arange(0, radio_map.shape[0]+1, 5),
                   np.arange(0, radio_map.shape[0]+1, 5))
        plt.yticks(np.arange(0, radio_map.shape[1]+1, 5),
                   np.arange(0, radio_map.shape[1]+1, 5))
        plt.legend(loc='lower left')
        plt.tight_layout()
        plt.savefig('rm.png', dpi=400)
        plt.show()


class Transmitter:
    '''Class to save a transmitter.
    
    Properties:
    tx_type : str
        Type of the transmitter. Can be 'tx' or 'jammer'.
    tx_pos : tuple
        Position of the transmitter in meters.
    tx_power : float
        Transmit power of the transmitter in dBm.
    '''

    def __init__(self, tx_type, tx_pos, tx_power):
        if tx_type not in ['tx', 'jammer']:
            raise ValueError('Type has to be either tx or jammer.')
        self.tx_type  = tx_type     # 'tx' or 'jammer'
        self.tx_pos   = tx_pos      # in meters
        self.tx_power = tx_power    # in dBm

    def __str__(self) -> str:
        return_string = 'Regular transmitter\n' if self.tx_type == 'tx' else 'Jammer\n'
        return_string += f'  tx_pos: ({self.tx_pos[0]:.2f} m, {self.tx_pos[1]:.2f} m)\n'
        return_string += f'  tx_power: {self.tx_power:.1f} dBm'

        return return_string    


def jammers_list_to_binary(jammers_list):
    '''Converts a list of jammers to a binary list,
    where each element is a 1 if there is a jammer in the corresponding radio map
    and 0 otherwise.

    jammers_list : list
        List of jammers in the original radio map.
    '''
    binary_list = np.zeros(len(jammers_list))
    for i in range(len(jammers_list)):
        if len(jammers_list[i]) > 0:
            binary_list[i] = 1
    
    return binary_list


def generate_measurement_points(method, shape, **kwargs):
    '''Generates the coordinates of measurement points for a given method (arrangement
    of the measurement points) and shape of the measurement area.

    method : str
        Method to generate the measurement points. Can be 'grid' or 'random'.
    shape : tuple
        Shape of the measurement area.
    kwargs : dict
        Additional arguments for the method.
        For method 'grid', the grid_size has to be provided.
        For 'custom1', the measurement points are placed at two opposite
        edges of the production lines. Attention: hard-coded!
    '''
    
    if method == 'grid':
        if 'grid_size' not in kwargs:
            raise ValueError('grid_size has to be in arguments.')
        grid_size = kwargs['grid_size']
        
        def calc_offset(length, grid_size):
            # If the length is not evenly dividable by the grid size, add an offset
            # to the grid to center the measurement points
            return np.floor((length - (np.floor(length/grid_size) * grid_size)) / 2).astype(int)

        x_offset = calc_offset(shape[0]-1, grid_size)
        y_offset = calc_offset(shape[1]-1, grid_size)

        meas_x = []
        meas_y = []
        # Currently, there is a workaround if the shape is evenly dividable by the grid size
        for x in range(x_offset, shape[0], grid_size):
            for y in range(y_offset, shape[1], grid_size):
                meas_x.append(x if x < shape[0] else x-1)
                meas_y.append(y if y < shape[1] else y-1)

        return meas_x, meas_y

    elif method == 'custom1':
        
        SU_coords1 = [[-14,9], [-8,9],[-3,9],[3,9],[8,9],[14,9],[-14,-9],[-8,-9],[-3,-9],[3,-9],[8,-9],[14,-9]]

        meas_x = []
        meas_y = []
        
        for val in SU_coords1:
            meas_x.append(val[0])
            meas_y.append(val[1])
        return meas_x,meas_y
  
    else:
        raise NotImplementedError()


def do_measurements(radiomap, meas_x, meas_y):
    '''Performs measurements at the given measurement points.

    radiomap : RadioMap
        Radio map to perform the measurements on.
    meas_x : list
        List of x coordinates of the measurement points.
    meas_y : list
        List of y coordinates of the measurement points.
    '''

    measurements = []

    for i in range(len(meas_x)):
        measurements.append(radiomap.radio_map[meas_x[i], meas_y[i]])

    return np.array(measurements)


def reshape_measurements_to_2D(meas_x, meas_y, shape, grid_size, measurements):
    '''Reshapes the measurements to a 2D array.

    meas_x : list
        List of x coordinates of the measurement points.
    meas_y : list
        List of y coordinates of the measurement points.
    shape : tuple
        Shape of the measurement area.
    grid_size : int
        Size of the grid.
    measurements : numpy.ndarray
        Measurements to reshape.
    '''

    def loc2idx(loc):
        return np.floor(loc/grid_size).astype(int)

    measurements_2D = np.zeros(shape)
    for i in range(len(meas_x)):
        measurements_2D[loc2idx(meas_x[i]), loc2idx(meas_y[i])] = measurements[i]

    return measurements_2D


def plot_radio_map_difference(rm_orig, rm_dt, plot_orig_tx=False, plot_dt_tx=False,
                              meas_x=[], meas_y=[], res_offset=0.5,
                              obstacle_mask=None):
    """Plots the difference between the original and the DT radio map.
    
    rm_orig : RadioMap
        Original radio map.
    rm_dt : RadioMap
        DT radio map.
    plot_orig_tx : bool
        If True, the true transmitter location is plotted.
    plot_dt_tx : bool
        If True, the location of the transmitters in the digital twin is plotted.
    meas_x : list
        List of x coordinates of the measurement points.
    meas_y : list
        List of y coordinates of the measurement points.
    res_offset : float
        Offset to match heatmap and transmitter locations.
    obstacle_mask : numpy.ndarray
        Mask for obstacles in the radio map. If provided, obstacle values are
        set to NaN.
    """

    plt.rcParams['text.usetex'] = latex_installed
    plt.rcParams['font.family'] = 'serif'

    # find maximum absolute value to later on center colorbar around 0
    vabs_max = np.max(np.abs(rm_orig.radio_map-rm_dt.radio_map))

    cbar_label = r'$P_\mathrm{rx} - \hat{P}_\mathrm{rx} [\mathrm{dB}]$' if latex_installed else 'P_rx - P^hat_rx [dB]'

    rm_diff = rm_orig.radio_map - rm_dt.radio_map
    if obstacle_mask is not None:
        rm_diff[obstacle_mask] = np.nan
    sns.heatmap(rm_diff.T, square=True, cbar=True, cmap='coolwarm',
                vmin=-vabs_max, vmax=vabs_max, cbar_kws={'label': cbar_label})
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')

    # Plot transmitter locations of original radio environment
    for transmitter in rm_orig.transmitters:
        plt.plot(transmitter.tx_pos[0]+res_offset, transmitter.tx_pos[1]+res_offset,
                  'v', color=COLOR_TX_ORIG, markersize=10)
    for jammer in rm_orig.jammers:
        plt.plot(jammer.tx_pos[0]+res_offset, jammer.tx_pos[1]+res_offset,
                 'X', markerfacecolor=COLOR_JAMMER_FACE,
                 markeredgecolor=COLOR_JAMMER_EDGE, markersize=10)
    
    # Plot transmitter locations of DT radio environment
    for transmitter in rm_dt.transmitters:
        plt.plot(transmitter.tx_pos[0]+res_offset, transmitter.tx_pos[1]+res_offset,
                 'v', markerfacecolor=COLOR_TX_DT_FACE, markeredgecolor=COLOR_TX_DT_EDGE,
                  markersize=10)

    # Plot measurement locations
    if len(meas_x) > 0:
        plt.scatter(np.array(meas_x)+res_offset, np.array(meas_y)+res_offset,
                    marker='.', color='black', label='Sensing unit (SU)')

    # Create legend entries
    if plot_orig_tx:
        plt.plot([], [], 'v', color=COLOR_TX_ORIG, markersize=10, label='True transmitter')
        if len(rm_orig.jammers) > 0:
            plt.plot([], [], 'X', markerfacecolor=COLOR_JAMMER_FACE, 
                     markeredgecolor=COLOR_JAMMER_EDGE, markersize=10,
                     label='Jammer')
    if plot_dt_tx:
        plt.plot([], [], 'v', markerfacecolor=COLOR_TX_DT_FACE,
                  markeredgecolor=COLOR_TX_DT_EDGE,
                  markersize=10, label='DT transmitter')

    if plot_orig_tx or plot_dt_tx or len(meas_x) > 0:
        plt.legend()

    plt.gca().invert_yaxis()
    plt.xticks(np.arange(0, rm_orig.radio_map.shape[0]+1, 5),
               np.arange(0, rm_orig.radio_map.shape[0]+1, 5))
    plt.yticks(np.arange(0, rm_orig.radio_map.shape[1]+1, 5),
               np.arange(0, rm_orig.radio_map.shape[1]+1, 5))
    plt.tight_layout()
    plt.savefig('rm_diff.png', dpi=400)
    plt.show()


def reshape_measurement_collection_to_2D(meas_x, meas_y, measurements1D):
    '''Reshapes a collection of measurements from 1D to 2D array.

    meas_x : list
        List of x coordinates of the measurement points.
    meas_y : list
        List of y coordinates of the measurement points.
    measurements1D : list
        List of measurements to reshape.
    '''
    shape = (len(np.unique(meas_x)), len(np.unique(meas_y)))
    sorted_meas_x = np.sort(np.unique(meas_x))
    grid_size = sorted_meas_x[1] - sorted_meas_x[0]

    measurements2D = []
    for i in range(len(measurements1D)):
        measurements2D.append(reshape_measurements_to_2D(meas_x, meas_y, shape, grid_size, measurements1D[i]))
        
    return np.array(measurements2D)
