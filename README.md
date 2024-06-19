# advancing-spectrum-anomaly-detection
Code for the paper "Advancing Spectrum Anomaly Detection through Digital Twins"

## Introduction

...


## Scenario Creation

The scenario for ray tracing is created using Blender and a Python script. The script `create_scenario_with_obstacles.py`is provided in the folder `blender-python`.  To execute the script, open Blender and got to the `Scripting` tab. Then, open the script and run it. The script will create a scenario with obstacles and save it in the folder `scenes`. The scene is exported in the `mitsuba` format which can be processed by `sionna`.

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

They are generated using the script `src\dataset_generation\pathloss_map_generation.py'`. The configuration is contained in the file `src\dataset_generation\conf\pathloss_map_generation.yaml`. Here, for example the scene number and the number of generated pathloss maps as well as the ray tracing parameters can be configured.