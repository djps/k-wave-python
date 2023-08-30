import numpy as np
import numpy.typing as npt

import os
import logging
import sys
from typing import List
import matplotlib.pyplot as plt
from skimage import measure
import meshio
import pyvista
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image

import nibabel as nib

# create logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
# create console and file handlers and set level to debug
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.DEBUG)
fh = logging.FileHandler(filename='runner_skull.log')
fh.setLevel(logging.DEBUG)
# create formatter
formatter = logging.Formatter('%(asctime)s | %(name)s | %(levelname)s | %(message)s')
# add formatter to ch, fh
ch.setFormatter(formatter)
fh.setFormatter(formatter)
# add ch, fh to logger
logger.addHandler(ch)
logger.addHandler(fh)
# propagate
ch.propagate = True
fh.propagate = True
logger.propagate = True

from kwave.data import Vector
from kwave.utils.kwave_array import kWaveArray
from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ksource import kSource
from kwave.ksensor import kSensor
from kwave.utils.signals import create_cw_signals
from kwave.utils.filters import extract_amp_phase
from kwave.kspaceFirstOrder3D import kspaceFirstOrder3DG

from kwave.options import SimulationOptions, SimulationExecutionOptions





# tussim_skull_3D(t1_filename, ct_filename, output_dir, focus_coords_in, bowl_coords_in, focus_depth):
#TUSSIM_SKULL_3D Run 3D k-wave acoustic simulation transcranially with skull
#   estimated using CT (or pseudo-CT) images.
#
# The transducer is modelled on the NeuroFUS CTX-500-4 with free-field Isppa
# of 20 W/cm^2. If you would like the simulation for a different free-field
# Isppa or a different 4-element transducer, you will have to provide your
# own source pressure and phase as optional Name-Value paired input arguments.
#
# The simulation grid is 256x256x256, with grid spacing determined by PPW
# (at recommended PPW = 6, grid spacing = 0.5x0.5x0.5mm^3).
#
# You will need to supply a co-registered T1-weighted MR image and CT (or
# pseudo-CT) image for use in the simulations. These images are recommended
# to be resampled to 1mm^3 isotropic voxels.
#
# Running the script without the acoustic or thermal simulation allows you
# to check that the transducer position relative to the head is correct.
# If you are satisfied, re-run the script with ('RunAcousticSim', true).
#
# Usage:
#   tussim_skull_3D(t1_filename, ct_filename, output_dir, ...
#       focus_coords, bowl_coords, focus_depth)
#   tussim_skull_3D(t1_filename, ct_filename, output_dir, ...
#       focus_coords, bowl_coords, focus_depth, Name, Value)
#   tussim_skull_3D('sub-test01_t1w.nii', 'sub-test01_pct.nii', 'output_dir', ...
#       [99, 161, 202], [90, 193, 262], 60, 'RunAcousticSim', true);
#
# Inputs:
#   t1_filename:    Full file path to the T1-weighted MR image.
#   ct_filename:    Full file path to the CT (or pseudo-CT) image.
#   output_dir:     Full path to the output directory.
#   focus_coords:   3-element array of voxel coordinates of the desired TUS
#                   focus. Add 1 if reading these off a viewer that uses
#                   zero-indexing (MATLAB indexing starts from 1).
#   bowl_coords:    3-element array of voxel coordinates of the centre of
#                   the transducer base. Add 1 if reading these off a
#                   viewer that uses zero-indexing.
#   focus_depth:    Distance from transducer face to intended focus in mm,
#                   rounded to the nearest integer.
#
# Optional Name-Value pair inputs:
#   'PPW':              Points per wavelength (default: 3, recommended: 6).
#   'RunAcousticSim':   Boolean controlling whether to run acoustic
#                       simulation (default: false).
#   'RunThermalSim':    Boolean controlling whether to run thermal
#                       simulation(default: false).
#   'PulseDur':         Pulse duration in s (default: 0.02 s).
#   'PulseRepInt':      Pulse repetition interval in s (default: 0.2s).
#   'PulseTrainDur':    Pulse train duration in s (default: 80 s).
#   'SourcePressure':   Source pressure in Pa.
#   'SourcePhase':      4-element array of phases of each transducer
#                       element in degrees for the focal depth required.
#   'RunCppCode':       Where to run the C++ simulation code (default: 'matlab')
#                       Options are 'matlab', 'terminal', 'linux_system'
#
# WARNING: Default acoustic values in function are for 500 kHz transducers.
# Please set your own values if using a transducer with a central frequency
# other than 500kHz.
#
# Dependencies:
#   k-Wave Toolbox v 1.4 (http://www.k-wave.org)
#     	Copyright (C) 2009-2017 Bradley Treeby
#
# Author: Siti N. Yaakub, University of Plymouth, 7 Sep 2022
#         (edited 2 Jun 2023)


verbose: bool = True
doPlotting: bool = True
savePlotting: bool = True

t1_filename: str = 'data/skull/sub-test01_t1w.nii'

ct_filename: str = 'data/skull/sub-test01_pct.nii'

output_dir: str = 'data/skull/'

# shift back by one for zero indexing
focus_coords_in: List[int] = [int(98), int(160), int(201)]

# shift back by one for zero indexing
bowl_coords_in: List[int] = [int(89), int(192), int(261)]

# Distance from transducer face to intended focus in mm, rounded to the nearest integer.
focus_depth: int = int(60)


# =========================================================================
# DEFINE THE MEDIUM PARAMETERS
# =========================================================================


# Cut-off values for CT data
hu_min: float = 300.0   # minimum skull HU
hu_max: float = 2000.0  # maximum skull HU, but changed

# medium parameters
c_min: float = 1500           # sound speed [m/s]
c_max: float = 3100           # max. speed of sound in skull (F. A. Duck, 2013.) [m/s]
rho_min: float = 1000         # density [kg/m^3]
rho_max: float = 1900         # max. skull density [kg/m3]
alpha_power: float = 1.43     # Robertson et al., PMB 2017 usually between 1 and 3? from Treeby paper
alpha_coeff_water: float = 0  # [dB/(MHz^y cm)] close to 0 (Mueller et al., 2017), see also 0.05 Fomenko et al., 2020?
alpha_coeff_min: float = 4    #
alpha_coeff_max: float = 8.7  # [dB/(MHz cm)] Fry 1978 at 0.5MHz: 1 Np/cm (8.7 dB/cm) for both diploe and outer tables

# sound speed [m/s]
c0: float = 1500.0

# density [kg/m^3]
rho0: float = 1000.0

# Robertson et al., PMB 2017
alpha_power: float = 1.43

# [dB/(MHz^y cm)] close to 0 (Mueller et al., 2017),
# see also 0.05 Fomenko et al., 2020
alpha_coeff: float = 0.0


# =========================================================================
# DEFINE THE TRANSDUCER SETUP
# =========================================================================

# single spherical transducer with four concentric elements, same pressure,
# but different phases in order to get a coherent focus

# name of transducer
transducer: str = 'CTX500'

# pressure [Pa]
pressure: float = 51590.0

# phase offsets [degrees]
phase: npt.NDArray[np.float64] = np.array([0.0, 319.0, 278.0, 237.0])

# bowl radius of curvature [m]
source_roc = 63.2e-3

# this has to be a list of lists with each list in the main list being the
# aperture diameters of the elements given an inner, outer pairs
diameters = [[0, 1.28],
             [1.3, 1.802],
             [1.82, 2.19],
             [2.208, 2.52]]

# the data was provided in inches, so is scaled to metres
scale = 0.0254
diameters = [[ scale * i for i in inner ] for inner in diameters]

# frequency [Hz]
freq = 500e3

# source pressure [Pa]
source_amp = np.squeeze(np.tile(pressure, [1, 4]))

# phase [rad]
source_phase = np.squeeze(np.array([np.deg2rad(phase), ]))


# =========================================================================
# DEFINE COMPUTATIONAL PARAMETERS
# =========================================================================

ppw_x = 3                   # number of points per wavelength along x-axis
ppw_y = 3                   # number of points per wavelength along y-axis
ppw_z = 3                   # number of points per wavelength along z-axis

record_periods = 3          # number of periods to record

cfl = 0.3                   # CFL number

ppp_x = round(ppw_x / cfl)  # compute points per period along x-axis
ppp_y = round(ppw_y / cfl)  # compute points per period along y-axis
ppp_z = round(ppw_z / cfl)  # compute points per period along z-axis

# Load CT image (nifti format)
# voxel size = 1 x 1 x 1 mm3, matrix size: varies
input_ct = nib.load(ct_filename).get_fdata()
# get header information
header_ct = nib.load(ct_filename).header

# Load MR image (nifti format)
input_t1 = nib.load(t1_filename).get_fdata()
header_t1 = nib.load(t1_filename).header

if verbose:
    print(header_ct)
    print(header_t1)

# calculate the grid spacing based on PPW and frequency
dx = c_min / (ppw_x * freq)  # in mm
dy = c_min / (ppw_y * freq)
dz = c_min / (ppw_z * freq)

print(dx)

# resample input images to grid res (iso)
scale_factor_x = round(header_ct['pixdim'][1] / (dx * 1e3), 2)
scale_factor_y = round(header_ct['pixdim'][2] / (dy * 1e3), 2)
scale_factor_z = round(header_ct['pixdim'][3] / (dz * 1e3), 2)

scale_factor = np.array([scale_factor_x, scale_factor_y, scale_factor_z])

# from skimage.transform import resize
# print(scale_factor, scale_factor_x, header_ct['pixdim'][1] / (dx * 1e3))
# print(type(input_ct[0]), type(input_ct[0, 0, 0]))

# ct_img = imresize3(input_ct, 'cubic', 'Scale', scale_factor);
# ct_img = np.array(Image.fromarray(input_ct).resize(208, 208))

# t1_img = imresize3(input_t1, 'cubic', 'Scale', scale_factor);
# t1_img = np.array(Image.fromarray(input_t1).resize(208, 208))

ct_img = np.array(input_ct, dtype=np.float32)
t1_img = np.array(input_t1, dtype=np.int16)

focus_coords = np.round(focus_coords_in * scale_factor).astype(int)
bowl_coords = np.round(bowl_coords_in * scale_factor).astype(int)

# update hu_max
ct_max = np.max(ct_img.flatten())
if (ct_max < hu_max):
    hu_max = ct_max

del ct_max

# truncate CT HU (see Marsac et al., 2017)
skull_model = ct_img
skull_model[skull_model < hu_min] = 0  # only use HU for skull acoustic properties
skull_model[skull_model > hu_max] = hu_max  # this does nothing

# centre grid at midpoint between transducer and focus coordinates
midpoint = np.round((bowl_coords + focus_coords) // 2.0).astype(int)

# pad images by 128 grid points on each side
padx: int = int(128)
pady: int = int(128)
padz: int = int(128)

padding = [padx, pady, padz]

# temporary padded container for model data which will be cropped
tmp_model = np.zeros((np.shape(skull_model)[0] + 2 * padx,
                      np.shape(skull_model)[1] + 2 * pady,
                      np.shape(skull_model)[2] + 2 * padz), dtype=np.float32)
# fill container with data
tmp_model[padx:np.shape(skull_model)[0] + padx,
          padx:np.shape(skull_model)[1] + pady,
          padx:np.shape(skull_model)[2] + padz] = skull_model

# shift the midpoint by padding value
tmp_midpoint = np.asarray(midpoint + padding).astype(int)

# get indices which define the centre by the midpoint
# grid size = 256x256x256, new midpoint coords = [128,128,128]
shift_idx = np.asarray([[tmp_midpoint[0] - padx, tmp_midpoint[0] + padx],
                        [tmp_midpoint[1] - pady, tmp_midpoint[1] + pady],
                        [tmp_midpoint[2] - padz, tmp_midpoint[2] + padz]], dtype=int)

# create model by cropped data
model = tmp_model[shift_idx[0, 0]:shift_idx[0, 1],
                  shift_idx[1, 0]:shift_idx[1, 1],
                  shift_idx[2, 0]:shift_idx[2, 1]]

shift_x: int = padx - midpoint[0]
shift_y: int = pady - midpoint[1]
shift_z: int = padz - midpoint[2]

shift = [shift_x, shift_y, shift_z]

bowl_coords = bowl_coords + shift    # centre of rear surface of transducer [grid points]
focus_coords = focus_coords + shift  # point on the beam axis of the transducer [grid points]

# move t1 image
new_t1 = np.zeros(np.shape(model), dtype=np.int16)
# new indices: [[x0, x1], [y0, y1], [z0, z1]]
idx1 = np.zeros((3, 2), dtype=int)

for ii in np.arange(3, dtype=int):
    # index on lower bounds
    h0: int = shift_idx[ii, 0] - padding[ii]
    if (h0 <= 0):
        idx1[ii, 0] = 0  # must be changed to 0 indexing
    elif (h0 > 0):
        idx1[ii, 0] = h0
    # index on upper bounds
    h1: int = shift_idx[ii, 1] - padding[ii]
    if (h1 <= np.shape(ct_img)[ii]):
        idx1[ii, 1] = h1
    elif (h1 > np.shape(ct_img)[ii]):
        idx1[ii, 1] = np.shape(ct_img)[ii]

idx2 = np.zeros((3, 2), dtype=int)

idx2[:, 0] = np.asarray(shift)

idx2[:, 1] = np.asarray(np.shape(ct_img)) + np.asarray(shift)

for ii in np.arange(0, 3, dtype=int):
    # index on lower bounds
    if (idx2[ii, 0] <= 0):
        idx2[ii, 0] = 0  # must be changed to 0 indexing
    # index on upper bounds
    if (idx2[ii, 1] > np.shape(model)[ii]):
        idx2[ii, 1] = np.shape(model)[ii]

new_t1[idx2[0, 0]:idx2[0, 1],
       idx2[1, 0]:idx2[1, 1],
       idx2[2, 0]:idx2[2, 1]] = t1_img[idx1[0, 0]:idx1[0, 1],
                                       idx1[1, 0]:idx1[1, 1],
                                       idx1[2, 0]:idx1[2, 1]]
t1_img = new_t1

del new_t1
del tmp_model

# assign medium properties for skull
# derived from CT HU based on Marsac et al., 2017 & Bancel et al., 2021
model_offset = 0.0
hu_offset = 0.0
density = rho_min + (rho_max - rho_min) * (model - model_offset) / (hu_max - hu_offset)

sound_speed = c_min + (c_max - c_min) * (density - rho_min) / (rho_max - rho_min)

alpha_coeff = alpha_coeff_min + (alpha_coeff_max - alpha_coeff_min) * (1.0 - (model - hu_min) / (hu_max - hu_min))**0.5

# assign medium properties for non-skull (brain, soft tissue, modelled as water)
density[model == 0] = rho_min
sound_speed[model == 0] = c_min
alpha_coeff[model == 0] = alpha_coeff_water

# Now create the medium object
medium = kWaveMedium(
    sound_speed=sound_speed,
    density=density,
    alpha_coeff=alpha_coeff,
    alpha_power=alpha_power
)


# =========================================================================
# DEFINE THE KGRID
# =========================================================================
# The grid is based on the size of the skull model.

# compute the size of the grid
Nx, Ny, Nz = np.shape(model)

grid_size_points = Vector([Nx, Ny, Nz])
grid_spacing_meters = Vector([dx, dy, dz])

# create the k-space grid
kgrid = kWaveGrid(grid_size_points, grid_spacing_meters)
if verbose:
    logger.info("done kWaveGrid")

# =========================================================================
# DEFINE THE TIME VECTOR
# =========================================================================

# compute corresponding time stepping
dt = 1.0 / (ppp_x * freq)

# dt_stability_limit = checkStability(kgrid, medium);
# if dt_stability_limit ~= Inf
#     dt = dt_stability_limit;
# end

# calculate the number of time steps to reach steady state
t_end = np.sqrt(kgrid.x_size**2 + kgrid.y_size**2) / c_min

# create the time array using an integer number of points per period
Nt = round(t_end / dt)
kgrid.setTime(Nt, dt)

# calculate the actual CFL after adjusting for dt
cfl = c_min * dt / np.max(np.asarray([dx, dy, dz]))

ppw = np.max(np.asarray([ppp_x, ppp_y, ppp_z]))

ppp = round(ppw / cfl)

# print details
if verbose:
    print('PPW = ' + str(ppw))
    print('CFL = ' + str(cfl))
    print('PPP = ' + str(ppp))

# =========================================================================
# DEFINE THE SOURCE PARAMETERS
# =========================================================================

# set bowl position and orientation
bowl_coords = np.asarray(bowl_coords).astype(int)
bowl_pos = [float(kgrid.x_vec[bowl_coords[0]]),
            float(kgrid.y_vec[bowl_coords[1]]),
            float(kgrid.z_vec[bowl_coords[2]])]

focus_coords = np.asarray(focus_coords).astype(int)
focus_pos = [float(kgrid.x_vec[focus_coords[0]]),
             float(kgrid.y_vec[focus_coords[1]]),
             float(kgrid.z_vec[focus_coords[2]])]

# create empty kWaveArray instance which will specify the transducer properties
karray = kWaveArray(bli_tolerance=0.01,
                    upsampling_rate=16,
                    single_precision=True)
if verbose:
    logger.info("done kWaveArray")

# add bowl shaped element
karray.add_annular_array(bowl_pos, source_roc, diameters, focus_pos)
if verbose:
    logger.info("done add_annular_array")

# create time varying source
source_sig = create_cw_signals(np.squeeze(kgrid.t_array),
                               freq,
                               source_amp,
                               source_phase)
if verbose:
    logger.info("done create_cw_signals")

# make a source object
source = kSource()
if verbose:
    logger.info("done kSource")

# assign binary mask using the karray
source.p_mask = karray.get_array_binary_mask(kgrid)
if verbose:
    logger.info("done get_array_binary_mask")

# assign source pressure output in time
source.p = karray.get_distributed_source_signal(kgrid, source_sig)
if verbose:
    logger.info("done get_distributed_source_signal")

# =========================================================================
# DEFINE THE SENSOR PARAMETERS
# =========================================================================

sensor = kSensor()
if verbose:
    logger.info("done kSensor")

# set sensor mask: the mask says at which points data should be recorded
sensor.mask = np.ones((Nx, Ny, Nz), dtype=bool)

# set the record type: record the pressure waveform
sensor.record = ['p']

# record the final few periods when the field is in steady state
sensor.record_start_index = kgrid.Nt - record_periods * ppp

# =========================================================================
# DEFINE THE SIMULATION PARAMETERS
# =========================================================================

input_filename = 'brics_skull_input.h5'
output_filename = 'brics_skull_output.h5'

DATA_CAST = 'single'
DATA_PATH = 'data/skull/'
BINARY_PATH = './kwave/bin/windows/'

# set input options
if verbose:
    logger.info("simulation_options")

# options for writing to file, but not doing simulations
simulation_options = SimulationOptions(
    data_cast=DATA_CAST,
    data_recast=True,
    save_to_disk=True,
    input_filename=input_filename,
    output_filename=output_filename,
    save_to_disk_exit=False,
    data_path=DATA_PATH,
    pml_inside=False)

if verbose:
    logger.info("execution_options")

execution_options = SimulationExecutionOptions(
    is_gpu_simulation=True,
    delete_data=False,
    verbose_level=2,
    binary_path=BINARY_PATH)


# =========================================================================
# RUN THE SIMULATION
# =========================================================================

if verbose:
    logger.info("kspaceFirstOrder3DG")

sensor_data = kspaceFirstOrder3DG(
    medium=medium,
    kgrid=kgrid,
    source=source,
    sensor=sensor,
    simulation_options=simulation_options,
    execution_options=execution_options)

# =========================================================================
# POST-PROCESS
# =========================================================================

# sampling frequency
fs = 1.0 / kgrid.dt

# get Fourier coefficients
amp, phi, f = extract_amp_phase(sensor_data, fs, freq, dim=1,
                                fft_padding=1, window='Rectangular')

# reshape data
p = np.reshape(amp, (Nx, Ny, Nz))

# extract pressure on beam axis
amp_on_axis = p[:, int(Ny / 2), int(Nz / 2)]

# define axis vectors for plotting
x_vec = kgrid.x_vec - kgrid.x_vec[0]
y_vec = kgrid.y_vec

# scale axes to mm
x_vec = np.squeeze(x_vec) * 1e3
y_vec = np.squeeze(y_vec) * 1e3

# scale pressure to MPa
amp_on_axis = amp_on_axis * 1e-6

# location of maximum pressure
max_pressure = np.max(p.flatten())
idx = np.argmax(p.flatten())
mx, my, mz = np.unravel_index(idx, np.shape(p))

# pulse length [s]
pulse_length = 20e-3

# pulse repetition frequency [Hz]
pulse_rep_freq = 5.0

if (model[mx, my, mz] > 1):
    Isppa = max_pressure**2 / (2 * np.max(medium.density.flatten()) * np.max(medium.sound_speed.flatten()) )  # [W/m2]
elif (model[mx, my, mz] == 0):
    Isppa = max_pressure**2 / (2 * rho_min * c_min)  # [W/m2]

Isppa = Isppa * 1e-4  # [W/cm2]
#     Ispta = Isppa * pulse_length * pulse_rep_freq # [W/cm2]

# MI (mechanical index) is max_pressure (in MPa) / sqrt freq (in MHz)
MI = max_pressure * 1e-6 / np.sqrt(freq * 1e-6)

# find -6dB focal volume
dB = -6
ratio = 10**(dB / 20.0) * max_pressure
tmp_focal_vol = np.where(p > ratio, p, 0.0)

# get largest connected component - probably the main focus
cc = measure.label(tmp_focal_vol)
print(np.shape(cc.label[0]), np.size(cc.label[0]))


verts, faces = measure.marching_cubes(p, ratio)
points = verts
cells = [("triangle", faces)]
mesh = meshio.Mesh(points, cells)
mesh.write("foo2.vtk")
# open to get volume
dataset2 = pyvista.read('foo2.vtk')
islands = dataset2.connectivity(largest=False)

dataset = pyvista.read('foo.vtk')
# print( dir(dataset) )
largest = dataset.connectivity(largest=True)

# cc = bwconncomp(tmp_focal_vol)

focal_vol = np.size(cc.label[0]) * (dx * 1e3)**3

del tmp_focal_vol
del cc

p_focus = p[focus_coords[0], focus_coords[1], focus_coords[2]]
isppa_focus = p_focus**2 / (2.0 * rho_min * c_min) * 1e-4

# Check whether max pressure point is in brain
if (np.linalg.norm(np.array(focus_coords) - np.array([mx, my, mz]) ) * dx * 1e3 > 5):
    raise ValueError("Maximum pressure point is more than 5 mm away from the intended focus. \nIt is likely that the maximum pressure is at the skull interface. \nPlease check output!")

    # # Create Plots
    # fig1 = plt.figure()
    # ax1 = fig.add_subplot(111, projection='3d')
    # im1 = ax1.imshow(np.rot90(np.squeeze(t1_img[mx, :, :]), vmin=50, vmax=500, cmap=gray, interpolation='none')
    # im2 = ax1.imshow(np.rot90(np.squeeze(p[mx, :, :])) * 1e-6, cmap=turbo, interpolation='none', alpha=0.5)
    # ax1.set_xticks([])
    # ax1.set_xticks([], minor=True)
    # ax1.set(title=r'Acoustic Pressure Amplitude')
    # ax1.grid(False)
    # cbar1 = fig1.colorbar(im1, ax=ax1)
    # cbar1.set_label('MPa')

    # saveas(gcf, fullfile(output_dir, 'pressure_sag.jpg'));

    # figure;
    # ax1 = axes;
    # imagesc(ax1, imrotate(squeeze(t1_img(mx,:,:)),90), [50,500]);
    # hold all;
    # ax2 = axes;
    # im2 = imagesc(ax2, imrotate(squeeze(p(mx,:,:)>(0.5*max_pressure))*1e-6,90));
    # im2.AlphaData = 0.5;
    # linkaxes([ax1,ax2]); ax2.Visible = 'off'; ax2.XTick = []; ax2.YTick = [];
    # colormap(ax1,'gray')
    # colormap(ax2,'turbo')
    # set([ax1,ax2],'Position',[.17 .11 .685 .815]);
    # cb2 = colorbar(ax2,'Position',[.85 .11 .0275 .815]);
    # xlabel(cb2, '[MPa]');
    # title(ax1,'50# Acoustic Pressure Amplitude')
    # saveas(gcf, fullfile(output_dir, 'pressure_sag_50#.jpg'));

    # figure;
    # ax1 = axes;
    # imagesc(ax1, imrotate(squeeze(t1_img(:,my,:)),90), [50,500]);
    # hold all;
    # ax2 = axes;
    # im2 = imagesc(ax2, imrotate(squeeze(p(:,my,:))*1e-6,90));
    # im2.AlphaData = 0.5;
    # linkaxes([ax1,ax2]); ax2.Visible = 'off'; ax2.XTick = []; ax2.YTick = [];
    # colormap(ax1,'gray')
    # colormap(ax2,'turbo')
    # set([ax1,ax2],'Position',[.17 .11 .685 .815]);
    # cb2 = colorbar(ax2,'Position',[.85 .11 .0275 .815]);
    # xlabel(cb2, '[MPa]');
    # title(ax1,'Acoustic Pressure Amplitude')
    # saveas(gcf, fullfile(output_dir, 'pressure_cor.jpg'));

    # figure;
    # ax1 = axes;
    # imagesc(ax1, imrotate(squeeze(t1_img(:,:,mz)),90), [50,500]);
    # hold all;
    # ax2 = axes;
    # im2 = imagesc(ax2, imrotate(squeeze(p(:,:,mz))*1e-6,90));
    # im2.AlphaData = 0.5;
    # linkaxes([ax1,ax2]); ax2.Visible = 'off'; ax2.XTick = []; ax2.YTick = [];
    # colormap(ax1,'gray')
    # colormap(ax2,'turbo')
    # set([ax1,ax2],'Position',[.17 .11 .685 .815]);
    # cb2 = colorbar(ax2,'Position',[.85 .11 .0275 .815]);
    # xlabel(cb2, '[MPa]');
    # title(ax1,'Acoustic Pressure Amplitude')
    # saveas(gcf, fullfile(output_dir, 'pressure_ax.jpg'));

    # ### Save pressure field and -6dB volume as nifti file
    # p_out = zeros(size(ct_img),'double');
    # p_out(idx1[0,0]:idx1[0,1], idx1[1,0]:idx1[1,1], idx1[2,0]:idx1[2,1]) = ...
    #     p(idx2[0,0]:idx2[0,1], idx2[1,0]:idx2[1,1], idx2[2,0]:idx2[2,1]);
    # p_out = imresize3(p_out, 'cubic', 'Scale', 1./scale_factor);
    # header.ImageSize = size(p_out);
    # header.Filename=[]; header.Filemoddate=[]; header.Filesize=[]; header.raw=[];
    # header.Datatype='double'; header.BitsPerPixel=32;
    # niftiwrite(p_out, fullfile(output_dir, 'pressure_field.nii'), header);

    # focal_vol_bin = int16(p_out > 0.5*max_pressure);
    # cc = bwconncomp(focal_vol_bin);
    # focal_vol_lcc = int16(zeros(size(p_out)));
    # focal_vol_lcc(cc.PixelIdxList{1}) = 1;
    # header.Datatype='int16'; header.BitsPerPixel=16;
    # niftiwrite(focal_vol_lcc, fullfile(output_dir, 'focal_volume_bin.nii'), header);

    # # find max pressure point on original image
    # [max_pressure, ~] = max(p_out(logical(focal_vol_lcc))); # [Pa]
    # idx = find(p_out==max_pressure);
    # [mx, my, mz] = ind2sub(size(p_out), idx);

    # ### Summary
    # # Print summary to command window
    # print('PPW = ' num2str(ppw))
    # print('CFL = ' num2str(cfl))
    # print('Coordinates of max pressure: [' num2str(mx) ', ' num2str(my) ', ' num2str(mz) ']')
    # print('Max Pressure = ' num2str(max_pressure * 1e-6) ' MPa')
    # print('MI = ' num2str(MI))
    # print('Isppa = ' num2str(Isppa) ' W/cm2')
    # print('Pressure at focus = ' num2str(p_focus * 1e-6) ' MPa');
    # print('Isppa at focus = ' num2str(isppa_focus) ' W/cm2')
    # print('-6dB focal volume = ' num2str(focal_vol) ' mm3')
    # print(' ')

    # # save output file in output_dir
    # clear source sensor_data;
    # save(fullfile(output_dir, 'acoustic_sim_output.mat'));

    # # Write summary to spreadsheet
    # # create result file if it does not exist and write header
    # if ~exist(fullfile(output_dir, 'simulation_results.csv'), 'file')
    #     disp('Result file does not exist, creating file.')
    #     fileID = fopen(fullfile(output_dir, 'simulation_results.csv'), 'w' );
    #     fprintf(fileID, '#s\n', ...
    #         ['Output directory, Focus coordinates, Bowl coordinates, Focus depth, ' ...
    #         'PPW, CFL, PPP, Coordinates of max pressure, ' ...
    #         'Max Pressure (MPa), MI, Isppa (W/cm2),' ...
    #         'Pressure at focus (MPa), Isppa at focus (W/cm2), ' ...
    #         '-6dB focal volume (mm3)']);
    #     fclose(fileID);
    # end

    # # write values
    # fileID = fopen(fullfile(output_dir, 'simulation_results.csv'),'a');
    # fprintf(fileID,['#s, #d #d #d, #d #d #d, #d, ' ...
    #     '#f, #f, #f, #d #d #d, ' ...
    #     '#f, #f, #f, #f, #f, #f\n'], ...
    #     output_dir, focus_coords_in, bowl_coords_in, focus_depth,  ...
    #     ppw, cfl, ppp, mx, my, mz, ...
    #     max_pressure * 1e-6, MI, Isppa, p_focus * 1e-6, isppa_focus, focal_vol);
    # fclose(fileID);
