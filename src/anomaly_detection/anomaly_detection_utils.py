import matplotlib.pyplot as plt
import numpy as np
import os
import pickle as pkl
import sys
import warnings

from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle


# Own modules
repo_name = 'advancing-spectrum-anomaly-detection'
module_path = __file__[:__file__.find(repo_name)+len(repo_name)]
sys.path.append(os.path.abspath(module_path))

from src.anomaly_detection import unsupervised_detection
from src.utils import radiomap_utils

def adjust_outlier_probability(X, y, p, num_samples_total=None, verbosity=0):
    """Given a dataset with outlier probability 0.5, the outlier probability is
    adjusted to p by removing samples from the anomaly class.
    
    X : array-like of shape (n_samples, n_features)
        The input samples.
    y : array-like of shape (n_samples,)
        The true labels.
    p : float
        The desired outlier probability.
    num_samples_total : int or None
        The total number of samples in the dataset. If None, all samples are used.
    verbosity : int
        Verbosity level.
        0: no output
        1: print number of samples and outlier probability
    """

    X_normal = X[y==0]
    y_normal = np.zeros(len(X_normal))

    if num_samples_total is not None:
        num_normal_samples = int(num_samples_total * (1-p))
        if num_normal_samples > len(X_normal):
            warnings.warn(f'Only {len(X_normal)} normal samples available,'\
                          f'but {num_normal_samples} are requested.' \
                          f'The number of normal samples is adjusted to {len(X_normal)}.')
            num_normal_samples = len(X_normal)
        elif num_normal_samples < 1:
            raise ValueError(f'Not enough normal samples available to adjust'\
                             ' the number of samples to {num_samples_total}.')

        X_normal = X_normal[:num_normal_samples]
        y_normal = y_normal[:num_normal_samples]

    nr_anomaly_samples = np.round(len(X_normal) * p / (1-p)).astype(int)
    np.random.seed(42) # Fixed seed!
    X_anomaly = np.random.permutation(X[y==1])[:nr_anomaly_samples]
    y_anomaly = np.ones(len(X_anomaly))
    X = np.concatenate((X_normal, X_anomaly))
    y = np.concatenate((y_normal, y_anomaly))

    if verbosity > 0:
        print(f'Test samples: {len(X)} -- Outlier probability: {len(X_anomaly) / len(X)*100:.2f}%')

    X, y = shuffle(X, y)
    
    return X, y


def analyze_neighbor_distances(X, noise_std=None, plot=False, percentile=90, n_neighbors=4):
    """Analyze the distances to the nearest neighbors.
    
    The method helps to find a suitable value for the epsilon parameter of DBSCAN.
    For each point in X, the distance to the nearest neighbor is calculated.
    
    X : array-like of shape (n_samples, n_features)
        The input samples.
    noise_std : float or None
        The shadowing noise standard deviation. Just required to add title to plot.
    plot : bool
        If True, the sorted distances are plotted.
    percentile : float
        The percentile of the distances to be used as epsilon.
    n_neighbors : int
        The number of neighbors to consider.
    """
    
    neigh = NearestNeighbors(n_neighbors=n_neighbors)
    nbrs = neigh.fit(X)
    distances, indices = nbrs.kneighbors(X)
    distances = np.sort(distances, axis=0)
    distances = distances[:,1]
    if plot:
        plt.plot(distances)
        if noise_std is not None:
            plt.title(f'Noise std: {noise_std} dB')
        plt.show()
    return np.percentile(distances, percentile)


def sionna_anomaly_detection(method,
                             outlier_probability,
                             scene_nr,
                             dataset_nr,
                             cfg,
                             noise_std,
                             probability=False, 
                             verbosity=0):
    """One run of anomaly detection for the given method and parameters for an sionna dataset.
    
    The dataset is loaded, preprocessed and split into train and test set.
    The anomaly detection is performed on the test set and the results are saved.
    
    method : str
        The anomaly detection method.
    outlier_probability : float
        The outlier probability.
    scene_nr : int
        Scene number.
    dataset_nr : int
        Dataset number
    cfg : OmegaConf
        Config object. Contains the parameters for the algorithms.
    noise_std : float
        Standard deviation of the noise added to measurements.
    probability : bool
        If True, the output of the anomaly detection is a "soft" value.
        Otherwise, the output is a binary classification (0 or 1).
    verbosity : int
        Verbosity level.
        0: no output, 1: some output
    """

    
    # Compile the filename of the measurement dataset for the given parameters
    meas_filename = os.path.join(module_path, 'datasets',
                                 f'scene{scene_nr}_measurements{dataset_nr}.pkl')

    # Load the dataset
    with open(meas_filename, 'rb') as fin:
        measurement_collection = pkl.load(fin)

    scaling = 'none' # 'none' or 'standard'
    if method not in ['dbscan', 'energy_detector', 'lof']:
        warnings.warn(f'Check if scaling is really not required for {method}!')

    # Dimension needs to be adjusted for ML measurement data set
    if len(measurement_collection.measurements_diff_list[0]) == 1:
        diff_list = np.array(measurement_collection.measurements_diff_list)[:,0,:]
    else:
        diff_list = np.array(measurement_collection.measurements_diff_list)


    # Add noise -----------------------------------------------------------------------------------
    
    diff_list += np.random.normal(0, noise_std, diff_list.shape)
    
    # Data preparation ----------------------------------------------------------------------------

    labels = radiomap_utils.jammers_list_to_binary(measurement_collection.jammers_list)

    if scaling == 'standard':
        scaler = StandardScaler()

    if scaling != 'none':
        measurements_scaled = scaler.fit_transform(diff_list)
    else:
        measurements_scaled = diff_list
    measurements_scaled = np.array(measurements_scaled)


    if cfg.sort_values:
        # Sort measurements for each sample by value to decrease variance
        measurements_scaled = np.flip(np.sort(measurements_scaled, axis=-1), axis=-1)

    # prepare n_kv-fold CV
    n_kv = 5
    kf = KFold(n_splits=n_kv, shuffle=True)

    # Cross validation
    for k, (train_idx, test_idx) in enumerate(kf.split(measurements_scaled, labels)):

        # Train and test set according to index of the run
        X_train = measurements_scaled[train_idx]
        y_train = labels[train_idx]
        X_test = measurements_scaled[test_idx]
        y_test = labels[test_idx]
        jammer_test = np.array(measurement_collection.jammers_list, dtype=object)[test_idx]

        # Train the model ---------------------------------------------------------------------------------

        if method == 'lof':
            model = unsupervised_detection.LocalOutlierFactorLearning(n_neighbors=cfg.lof.n_neighbors,
                                                                      probability=probability)
        elif method == 'energy_detector':
            model = unsupervised_detection.EnergyDetector(percentile=0.95, probability=probability)
        else:
            raise NotImplementedError(f'Not implemented method: {method}')

        if method in unsupervised_detection.unsupervised_methods:
            model.fit(X_train[y_train==0])

        # Test the model ----------------------------------------------------------------------------------

        # adjust outlier probability
        X_test, y_test = adjust_outlier_probability(X_test, y_test, p=outlier_probability,
                                                    num_samples_total=cfg.num_test_samples,
                                                    verbosity=verbosity)
        y_hat = model.predict(X_test)

        # Rescaling not necessary, as the task is a classification problem

        if not probability and method == 'lof':
            # Output of certain classifiers is 1 for inliers and -1 for outliers
            # Conversion required to apply F1 score
            y_hat[y_hat == 1] = 0
            y_hat[y_hat == -1] = 1

        if not probability and verbosity > 0:
            print(f'F1 Score: {f1_score(y_test, y_hat):.2f}')


        # Save the results --------------------------------------------------------------------------------

        # Name of the results file
        if probability:
            results_filenname = os.path.join(module_path, 'datasets', 'results', f'sionna_{dataset_nr}_prob.pkl')
        else:
            results_filenname = os.path.join(module_path, 'datasets', 'results', f'sionna_{dataset_nr}.pkl')

        # check if results already exists for this dataset
        if os.path.exists(results_filenname):
            # if so, load them
            with open(results_filenname, 'rb') as f_results:
                results = pkl.load(f_results)
        else:
            results = {}

        if method not in results:
            results[method] = {}    # Save results in dict, key is noise_std

        # save the results file with the results from this run added
        if k == 0:
            # save results from first run
            results[method][noise_std] = {'y_test': y_test, 'y_hat': y_hat, 'jammer': jammer_test}
        else:
            # append results from second run
            results[method][noise_std]['y_test'] = np.concatenate((results[method][noise_std]['y_test'], y_test))
            results[method][noise_std]['y_hat'] = np.concatenate((results[method][noise_std]['y_hat'], y_hat))
            results[method][noise_std]['jammer'] = np.concatenate((results[method][noise_std]['jammer'], jammer_test))

        with open(results_filenname, 'wb') as f_results:
            pkl.dump(results, f_results)

        