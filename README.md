# advancing-spectrum-anomaly-detection
Code for the paper "Advancing Spectrum Anomaly Detection through Digital Twins"

## Introduction

...


## Scenario Creation

The scenario for ray tracing is created using Blender and a Python script. The script `create_scenario_with_obstacles.py`is provided in the folder `blender-python`.  To execute the script, open Blender and got to the `Scripting` tab. Then, open the script and run it. The script will create a scenario with obstacles and save it in the folder `scenes`. The scene is exported in the `mitsuba` format which can be processed by `sionna`.

Adjustments to the scenario can be specified in the file `blender-python\conf\scene_attributes.yaml`. Please note, that the properties `transmitters`, `jammers` and `base_stations` are only used for visualization purposes.

To visualize the scenario, the script `create_scenario_visualization.py` can be used. The script runs creates the scenario and visualizes in addition the radio devices. After the script has run, select the camera and click with the right mouse button on the camera. Select `Set camera active` and afterwards press `F12`to render the scene. For rendering, `Eevee` provides fast results, whereas the results with `Cycles` are more realistic. When finished, the file needs to be saved manually.
