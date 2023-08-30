import h5py
from scipy.io import loadmat
import numpy as np
from typing import Any
import os
import subprocess
import platform
import socket
import warnings
from datetime import datetime
from typing import Optional
from post_process import PostProcess

verbose = False

with h5py.File("data/skull/tus_skull_input.h5", "r") as f2:
    # Print all root level object names (aka keys)
    # these can be group or dataset names
    if verbose:
        print("From GitHub")
        print("Keys: %s" % f2.keys())
    github = np.squeeze(np.array(f2["p_source_input"]))
    if verbose:
        print(np.shape(github))

# subprocess.run("./kwave/bin/windows/kspaceFirstOrder-CUDA.exe -i data/skull/tus_skull_input.h5 -o ./data/skull/tus_skull_output.h5 -s 1757 --verbose 2")

filename = "data/skull/tus_skull_output.h5"
freq = 500e3  # must be in Hz

data = PostProcess(filename, freq)
sensor_data = data.hdf5_loader(filename)

p = data.get_temporal_average(sensor_data)

data.plot3D(p)

# integration_points = makeCartRect(...
#                         obj.affine(obj.elements{element_num}.position), ...
#                         obj.elements{obj.number_elements}.length, ...
#                         obj.elements{obj.number_elements}.width, ...
#                         obj.elements{obj.number_elements}.orientation, ...
#                         m_integration);
# However I think this is incorrect, because now you only take the last orientation? Shouldn't it be:

# integration_points = makeCartRect(...
#                         obj.affine(obj.elements{element_num}.position), ...
#                         obj.elements{obj.number_elements}.length, ...
#                         obj.elements{obj.number_elements}.width, ...
#                         obj.elements{element_num}.orientation, ...
#                         m_integration);
# I don't know if I need to use affine() as well?
