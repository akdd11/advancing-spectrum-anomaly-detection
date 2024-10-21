# Advancing Spectrum Anomaly Detection through Digital Twins

## Introduction

The repository contains the code to run the simulations used for the paper "Advancing Spectrum Anomaly Detection through Digital Twins" which can be found [here](https://www.techrxiv.org/users/775914/articles/883996-advancing-spectrum-anomaly-detection-through-digital-twins). In the paper, a novel approach for identifying spectrum anomalies is proposed, which employs a digital twin of the radio environment. The simulations are based on ray tracing using [Sionna](https://nvlabs.github.io/sionna/). [Blender](https://www.blender.org/) is used to create the model of the scenario.

If you use the code or parts of it, please cite
```bibtex
@article{schoesser24,
title={Advancing Spectrum Anomaly Detection through Digital Twins},
DOI={10.36227/techrxiv.171470424.41432460/v2},
publisher={TechRxiv},
author={Schösser, Anton and Burmeister, Friedrich and Schulz, Philipp and Khursheed, Mohd Danish and Ma, Sinuo and Fettweis, Gerhard},
year={2024},
month=sep
}
```

### Module Versions

The code was developoed Python modules 3.10.10 and the following module versions have been used:
* Numpy: 1.23.5
* TensorFlow: 2.13.0
* Mitsuba: 3.2.1
* Sionna: 0.15.1
* Drjit: 0.4.1
* Hydra: 1.3.2


## Scenario Creation

For successful export of the scene, the [Mitsuba Add-on for Blender](https://github.com/mitsuba-renderer/mitsuba-blender) needs to be installed first. The scenario for ray tracing is created using Blender and a Python script. The script `create_scenario_with_obstacles.py`is provided in the folder `blender-python`.  To execute the script, open Blender and got to the `Scripting` tab. Then, open the script and run it. The script will create a scenario with obstacles and save it in the folder `scenes`. The scene is exported in the `mitsuba` format which can be processed by `sionna`.

Adjustments to the scenario can be specified in the file `blender-python\conf\scene_attributes.yaml`. Please note, that the properties `transmitters`, `jammers` and `base_stations` are only used for visualization purposes.

To visualize the scenario, the script `create_scenario_visualization.py` can be used. The script runs creates the scenario and visualizes in addition the radio devices. After the script has run, select the camera and click with the right mouse button on the camera. Select `Set camera active` and afterwards press `F12` to render the scene. For rendering, `Eevee` provides fast results, whereas the results with `Cycles` are more realistic. When finished, the file needs to be saved manually.

Please note, that the texture files are not part of this repository. They are only needed for visualization purpose, but not for the ray tracing. If needed, please download
them:
* [Concrete](https://www.poliigon.com/texture/poured-concrete-floor-texture/7656)
* [Metal](https://www.poliigon.com/texture/lightly-worn-galvanised-steel-industrial-metal-texture/3129)

and insert the files in the folders `blender-python\textures\ConcretePoured` and `blender-python\textures\MetalGalvanizedSteelWorn`, respectively.

## Dataset Generation

Generating the dataset to process consists of three steps. In a first step, a huge number of path loss maps for random transmitter locations are generated using ray tracing. Secondly, according to the number of legitimate transmitters and jammers and their associated transmit powers, they are combined (the powers are added up) to obtain the physical twin (PT) radio maps. In a third step, for each sample created in the last step, for each legitimate transmitter, the original locations of the legitimate transmitters are slightly assigned with a random offset to mimic localization inaccuracy and new pathloss maps are generated. The new resulting pathloss maps are then combined to obtain the digital twin (DT) radio maps. 

### Pathloss Map Generation

They are generated using the script `src\dataset_generation\pathloss_map_generation.py'`. The configuration is contained in the file `src\dataset_generation\conf\pathloss_map_generation.yaml`. Here, for example the scene number and the number of generated pathloss maps as well as the transmitter and receiver heights and ray tracing parameters can be configured.

### Radio Map Generation

In this step, the pathloss maps are combined to obtain the PT (original) which might also contain the jammer. To execute the radio map generation, run the script `src\dataset_generation\radio_map_generation.py`. The configuration is contained in the file `src\dataset_generation\conf\radio_map_generation.yaml`. Here, for example the scene number and the number of generated radio maps as well as the number and transmit powers of the transmitters and jammers can be configured.

### Measurement Generation

The last step of the dataset generation is denoted as measurement generation. In this step, for each PT radio map which was created in the previous step, its DT counterpart is generated. Therefore, the orginial  locations of the legitimate transmitters is shifted by a random offset to mimick localization inaccuracy. Moreover, the jammer is not known by the DT and therefore not modeled. Two approaches are compared in this work to generate the DT. The script is located at `src\dataset_generation\measurement_generation.py`.

#### Approach 1: Ray Tracing
The original scene file is used together with the shifted transmitter locations to generate the DT radio maps. From those, the estimated path loss values between the estimated transmitter location and the sensing units is extracted.

#### Approach 2: Machine Learning
New path loss maps are generated to train an ML model to estimate the path loss values between the estimated transmitter location and the sensing units. After the model is trained on the newly generated path loss maps, the model is used to infer the path loss values between the estimated transmitter location and the sensing units.

Currently, random forest (RF) is used as a multi-output regressor, i.e., for each sensing unit a separate RF model is trained.

As input, either only the estimated transmitter location coordinates can be used or additionally the distance between the estimated transmitter location and the sensing unit can be used. The second case is applied in the paper, as it slightly reduces the estimation error.

#### Dataset
After those steps are executed, the final dataset file is generate, which contains for eeach sensing unit the difference between the actual received power from the PT and the estimated received power from the DT as well as a binary label indicating whether a jammer was present or not.

#### File Numbering
Currently, the following number is used to identify the different scenarios:
* PL dataset:
  * 0: Sensing unit height: 1.5m, Transmitter height: 1.5m, used for PT generation
  * 1: Sensing unit height: 1.5m, Transmitter height: 1.5m, used for train the ML model for path loss estimation
  * 2: Sensing unit height: 5.0m, Transmitter height: 1.5m, used for PT generation
  * 3: Sensing unit height: 5.0m, Transmitter height: 1.5m, used for train the ML model for path loss estimation
* RM dataset:
  * 0: Sensing unit height: 1.5m, Transmitter height: 1.5m, used for PT generation
  * 2: Sensing unit height: 5.0m, Transmitter height: 1.5m, used for PT generation
* Measurement dataset:
  * 0: Sensing unit height: 1.5m, Transmitter height: 1.5m, using ray tracing to create the DT
  * 1: Sensing unit height: 1.5m, Transmitter height: 1.5m, using ML to create the DT
  * 2: Sensing unit height: 5.0m, Transmitter height: 1.5m, using ray tracing to create the DT
  * 3: Sensing unit height: 5.0m, Transmitter height: 1.5m, using ML to create the DT

The file number needs to be adjusted in the file `src\dataset_generation\conf\measurement_generation.yaml`.


# Anomaly Detection

Evaluating different anomaly detection algorithms on the generated datasets can be done with the script `src\anomaly_detection\sionna_anomaly_detection.py`. The configuration can be done with the configuration file  `src\anomaly_detection\conf\sionna_anomaly_detection.yaml`. Particularly, the dataset and the employed algorithm can be specified there. The results are saved in the folder `datasets\results`. To draw the ROC curves, the output of the anomaly detection must be a soft output, therefore `probability` must be set to `True` in the configuration file.

The ROC curves can be drawn then in the notebook `notebooks\roc_curves_sionna.ipynb`.
