import os
import sys
import numpy as np
from tqdm import tqdm
import pickle as pkl
import pandas as pd
from scipy.spatial import distance

#  Speed up sklearn
from sklearnex import patch_sklearn
patch_sklearn()

from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler,StandardScaler

repo_name = 'advancing-spectrum-anomaly-detection'
module_path = __file__[:__file__.find(repo_name)+len(repo_name)]
sys.path.append(os.path.abspath(module_path))


def generate_df(meas_x, meas_y, scene_nr, dataset_nr, measurement_method, samples_nr, feature_ext=True, save_dataset=False):
    '''Generates dataset suitable for supervised ML model training.
    meas_x: list
        list of x-coordinates of SUs.
    meas_y: list
        list of y-coordinates of SUs.
    scene_nr: int
        scene number.
    dataset_nr: int
        dataset number of the path loss dataset for learning the ML model.
    measurement_method: String
        Type of SU placement in the scene.
    samples_nr: int
        no of samples.
    feature_ext: Bool
        If True, dataset will have distance between SU and transmitter as a feature.
    save_dataset: Bool
        If True, dataset will be saved as CSV file.        
    '''
    filename = f'scene{scene_nr}_PLdataset{dataset_nr}.pkl'
    load_path = module_path + '\\' + 'datasets' + '\\' + filename    
    if measurement_method == 'custom1': 
        cols = ['tx_x', 'tx_y', 'su_0', 'su_1', 'su_2', 'su_3', 'su_4', 'su_5', 'su_6', 'su_7', 'su_8', 'su_9', 'su_10', 'su_11']
        cols_1= ['tx_x', 'tx_y','dist_0','dist_1','dist_2','dist_3','dist_4','dist_5','dist_6','dist_7','dist_8','dist_9','dist_10',
                'dist_11', 'su_0', 'su_1', 'su_2', 'su_3', 'su_4', 'su_5', 'su_6', 'su_7', 'su_8', 'su_9', 'su_10', 'su_11']
    else:
        raise NotImplementedError("Other measurement methods not implemented")

    if feature_ext:
        save_filename = f'scene{scene_nr}_pl_for_ML_{dataset_nr}_fe.csv'
        save_path = module_path + '\\' + 'datasets\\supervised learning' + '\\' + save_filename
        col_names = cols_1
    else: 
        save_filename = f'scene{scene_nr}_pl_for_ML_{dataset_nr}.csv'
        save_path = module_path + '\\' + 'datasets\\supervised learning' + '\\' + save_filename
        col_names = cols

    if os.path.isfile(os.path.join('', save_path)) and save_dataset:
        print(f'Dataset {save_filename} already exists. Overwrite? (y/n)')
        answer = input()
        if answer == 'y':
            os.remove(os.path.join('',save_path))
        else:
            sys.exit()
            
    with open(load_path, 'rb') as fin:
        plmc = pkl.load(fin)

    df_list = []
    
    for sample_id in tqdm(range(samples_nr)):
        plm_df = np.array(plmc.pathlossmaps[sample_id].tx_pos[:2])
        pathloss_values = [plmc.pathlossmaps[sample_id].pathloss[int(meas_x[i]), int(meas_y[i])] for i in range(len(meas_x))]
        if feature_ext:
            dist_list = [distance.euclidean(plm_df,[k,l]) for k,l in zip(meas_x,meas_y)]
            plm_df = np.append(plm_df, dist_list)
        plm_df = np.append(plm_df, pathloss_values)
        df_list.append(plm_df)

    # Convert the list of arrays to a NumPy array
    df_array = np.vstack(df_list)
    
    df = pd.DataFrame(df_array, columns=col_names)           
        
    if save_dataset:
        df.to_csv(save_path,index=False)    
    
    return df


def train_model(df, validate=False, algo='mor'):
    '''Prepares dataset and trains model.

    Input
    -----
    df : Pandas Dataframe
        Dataset containing pathloss values for each SU.
    validate : Bool
        If True, model will validated using a test dataset and RMSE will be calculated.
    algo : String
        Type of ML model to implement
    
    Output
    ------
    model : ML model
        Trained ML model.
        Currently only MultiOutputRegressor based on RandomForestRegressor is implemented.
    scaler_x : Scaler
        Scaler for features
    scaler_y : Scaler
        Scaler for target variables
    '''
    
    scaler_x = MinMaxScaler()     #Type of scaling for features
    scaler_y = StandardScaler()   #Type of scaling for target variables
    RANDOM_SEED = 42
    
    df.reset_index(drop=True,inplace=True)

    df.replace(np.inf, df.quantile(0.99), inplace=True)
    df.replace(np.nan, df.quantile(0.99), inplace=True)

    X = df.iloc[:,:14]          #feature limit is according to feature_ext
    y = df.iloc[:,14:]

    X_train,X_test,y_train, y_test = train_test_split(X,y, test_size=0.1, random_state=RANDOM_SEED)

    if algo == 'mor':
        if scaler_x is not None:
            X_train = scaler_x.fit_transform(X_train.values)
            X_test = scaler_x.transform(X_test.values)
        if scaler_y is not None:
            y_train = scaler_y.fit_transform(y_train.values)
            
        model = MultiOutputRegressor(RandomForestRegressor(n_estimators=400,min_samples_split=2, n_jobs=None), n_jobs=None)
        model.fit(X_train,y_train)
        print("Training completed")
        
        if validate:
            y_pred = model.predict(X_test)
            y_pred = scaler_y.inverse_transform(y_pred)
            score = mean_squared_error(y_test,y_pred, squared=False)
            print(f"Loss (RMSE): {score}")
        
        return model, scaler_x, scaler_y
    else:
        raise NotImplementedError("This ML approach is not implemented.")

    
def prepare_input(tx_pos_est, meas_x, meas_y, scaler_x, measurement_method):
    '''Converts tx_pos_est into ML model suitable format
    tx_pos_est: NumPy NdArray
        X,y,z coordinates of transmitter.
    meas_x: list
        list of x-coordinates of SUs.
    meas_y: list
        list of y-coordinates of SUs.
    measurement_method: String
        Type of SU placement in the scene.
    '''
    tx_input = tx_pos_est[0:2]
    dist_list = []
    if measurement_method == 'custom1':
        dist_list = [distance.euclidean(tx_pos_est[0:2],[k,l]) for k,l in zip(meas_x,meas_y)]
    elif measurement_method == 'grid':
        pass
    else:
        raise NotImplementedError("Other measurement methods are not implemented")
    
    tx_input+=dist_list
    tx_input = scaler_x.transform(np.asarray(tx_input).reshape(1,-1))
    
    return tx_input
