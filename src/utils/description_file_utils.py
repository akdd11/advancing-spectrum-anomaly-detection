import os
from scanf import scanf
import sys

repo_name = "advancing-spectrum-anomaly-detection"
module_path = __file__[: __file__.find(repo_name) + len(repo_name)]
sys.path.append(os.path.abspath(module_path))


def get_config_from_file(path_to_file):
    """Reads the config from a file and returns it as a dictionary.

    path_to_file: str
        Path to the text file containing the config.
    """
    config = {}
    with open(path_to_file, "r") as f_in:
        lines = f_in.readlines()

    def read_line_with_array(l):
        splitted_line = l.split(":")  # split into key and array
        splitted_line[1] = splitted_line[1].translate(
            str.maketrans("", "", "[]\n ")
        )  # only comma remains to separate the values
        config[splitted_line[0]] = [
            float(x) for x in splitted_line[1].split(",")
        ]  # comma-separated values to array

    for l in lines:

        if "scene_size" in l:
            read_line_with_array(l)
            continue

        A = scanf("%s: %s", l)
        if A == None:
            continue

        # if possible, convert value to float
        try:
            config[A[0]] = float(A[1])
        except ValueError:
            config[A[0]] = A[1]

    return config
