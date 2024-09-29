import numpy as np
from scipy.interpolate import interpn
from tqdm import tqdm
from typing import Union
from copy import deepcopy
import logging

from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ksensor import kSensor
from kwave.ksource import kSource
from kwave.kWaveSimulation import kWaveSimulation

from kwave.ktransducer import NotATransducer

from kwave.utils.conversion import db2neper
from kwave.utils.data import scale_time
from kwave.utils.filters import gaussian_filter
from kwave.utils.pml import get_pml
from kwave.utils.signals import reorder_sensor_data
from kwave.utils.tictoc import TicToc
from kwave.utils.dotdictionary import dotdict

from kwave.options.simulation_options import SimulationOptions

from kwave.kWaveSimulation_helper import extract_sensor_data, reorder_cuboid_corners, save_intensity

def pstd_elastic_3d(kgrid: kWaveGrid,
                    source: kSource,
                    sensor: Union[NotATransducer, kSensor],
                    medium: kWaveMedium,
                    simulation_options: SimulationOptions):
    """
    3D time-domain simulation of elastic wave propagation.

    DESCRIPTION:

        pstd_elastic_3d simulates the time-domain propagation of elastic waves
        through a three-dimensional homogeneous or heterogeneous medium given
        four input structures: kgrid, medium, source, and sensor. The
        computation is based on a pseudospectral time domain model which
        accounts for viscoelastic absorption and heterogeneous material
        parameters. At each time-step (defined by kgrid.dt and kgrid.Nt or
        kgrid.t_array), the wavefield parameters at the positions defined by
        sensor.mask are recorded and stored. If kgrid.t_array is set to
        'auto', this array is automatically generated using the makeTime
        method of the kWaveGrid class. An anisotropic absorbing boundary
        layer called a perfectly matched layer (PML) is implemented to
        prevent waves that leave one side of the domain being reintroduced
        from the opposite side (a consequence of using the FFT to compute the
        spatial derivatives in the wave equation). This allows infinite
        domain simulations to be computed using small computational grids.

        An initial pressure distribution can be specified by assigning a
        matrix of pressure values the same size as the computational grid to
        source.p0. This is then assigned to the normal components of the
        stress within the simulation function. A time varying stress source
        can similarly be specified by assigning a binary matrix (i.e., a
        matrix of 1's and 0's with the same dimensions as the computational
        grid) to source.s_mask where the 1's represent the grid points that
        form part of the source. The time varying input signals are then
        assigned to source.sxx, source.syy, source.szz, source.sxy,
        source.sxz, and source.syz. These can be a single time series (in
        which case it is applied to all source elements), or a matrix of time
        series following the source elements using MATLAB's standard
        column-wise linear matrix index ordering. A time varying velocity
        source can be specified in an analogous fashion, where the source
        location is specified by source.u_mask, and the time varying input
        velocity is assigned to source.ux, source.uy, and source.uz.

        The field values are returned as arrays of time series at the sensor
        locations defined by sensor.mask. This can be defined in three
        different ways. (1) As a binary matrix (i.e., a matrix of 1's and 0's
        with the same dimensions as the computational grid) representing the
        grid points within the computational grid that will collect the data.
        (2) As the grid coordinates of two opposing corners of a cuboid in
        the form [x1 y1 z1 x2 y2 z2]. This is equivalent to using a
        binary sensor mask covering the same region, however, the output is
        indexed differently as discussed below. (3) As a series of Cartesian
        coordinates within the grid which specify the location of the
        pressure values stored at each time step. If the Cartesian
        coordinates don't exactly match the coordinates of a grid point, the
        output values are calculated via interpolation. The Cartesian points
        must be given as a 3 by N matrix corresponding to the x, y, and z
        positions, respectively, where the Cartesian origin is assumed to be
        in the center of the grid. If no output is required, the sensor input
        can be replaced with `None`.

        If sensor.mask is given as a set of Cartesian coordinates, the
        computed sensor_data is returned in the same order. If sensor.mask is
        given as a binary matrix, sensor_data is returned using MATLAB's
        standard column-wise linear matrix index ordering. In both cases, the
        recorded data is indexed as sensor_data(sensor_point_index,
        time_index). For a binary sensor mask, the field values at a
        particular time can be restored to the sensor positions within the
        computation grid using unmaskSensorData. If sensor.mask is given as a
        list of cuboid corners, the recorded data is indexed as
        sensor_data(cuboid_index).p(x_index, y_index, z_index, time_index),
        where x_index, y_index, and z_index correspond to the grid index
        within the cuboid, and cuboid_index corresponds to the number of the
        cuboid if more than one is specified.

        By default, the recorded acoustic pressure field is passed directly
        to the output sensor_data. However, other acoustic parameters can
        also be recorded by setting sensor.record to a cell array of the form
        {'p', 'u', 'p_max', }. For example, both the particle velocity and
        the acoustic pressure can be returned by setting sensor.record =
        {'p', 'u'}. If sensor.record is given, the output sensor_data is
        returned as a structure with the different outputs appended as
        structure fields. For example, if sensor.record = {'p', 'p_final',
        'p_max', 'u'}, the output would contain fields sensor_data.p,
        sensor_data.p_final, sensor_data.p_max, sensor_data.ux,
        sensor_data.uy, and sensor_data.uz. Most of the output parameters are
        recorded at the given sensor positions and are indexed as
        sensor_data.field(sensor_point_index, time_index) or
        sensor_data(cuboid_index).field(x_index, y_index, z_index,
        time_index) if using a sensor mask defined as cuboid corners. The
        exceptions are the averaged quantities ('p_max', 'p_rms', 'u_max',
        'p_rms', 'I_avg'), the 'all' quantities ('p_max_all', 'p_min_all',
        'u_max_all', 'u_min_all'), and the final quantities ('p_final',
        'u_final'). The averaged quantities are indexed as
        sensor_data.p_max(sensor_point_index) or
        sensor_data(cuboid_index).p_max(x_index, y_index, z_index) if using
        cuboid corners, while the final and 'all' quantities are returned
        over the entire grid and are always indexed as
        sensor_data.p_final(nx, ny, nz), regardless of the type of sensor
        mask.

        pstd_elastic_3d may also be used for time reversal image reconstruction
        by assigning the time varying pressure recorded over an arbitrary
        sensor surface to the input field sensor.time_reversal_boundary_data.
        This data is then enforced in time reversed order as a time varying
        Dirichlet boundary condition over the sensor surface given by
        sensor.mask. The boundary data must be indexed as
        sensor.time_reversal_boundary_data(sensor_point_index, time_index).
        If sensor.mask is given as a set of Cartesian coordinates, the
        boundary data must be given in the same order. An equivalent binary
        sensor mask (computed using nearest neighbour interpolation) is then
        used to place the pressure values into the computational grid at each
        time step. If sensor.mask is given as a binary matrix of sensor
        points, the boundary data must be ordered using matlab's standard
        column-wise linear matrix indexing - this means, Fortran ordering.

    USAGE:
        sensor_data = pstd_elastic_3d(kgrid, medium, source, sensor, options)

    INPUTS:
    The minimum fields that must be assigned to run an initial value problem
    (for example, a photoacoustic forward simulation) are marked with a *.

        kgrid*                 - k-Wave grid object returned by kWaveGrid
                                containing Cartesian and k-space grid fields
        kgrid.t_array*         - evenly spaced array of time values [s] (set
                                to 'auto' by kWaveGrid)

        medium.sound_speed_compression*
                                - compressional sound speed distribution
                                within the acoustic medium [m/s]
        medium.sound_speed_shear*
                                - shear sound speed distribution within the
                                acoustic medium [m/s]
        medium.density*        - density distribution within the acoustic
                                medium [kg/m^3]
        medium.alpha_coeff_compression
                                - absorption coefficient for compressional
                                waves [dB/(MHz^2 cm)]
        medium.alpha_coeff_shear
                                - absorption coefficient for shear waves
                                [dB/(MHz^2 cm)]

        source.p0*             - initial pressure within the acoustic medium
        source.sxx             - time varying stress at each of the source
                                positions given by source.s_mask
        source.syy             - time varying stress at each of the source
                                positions given by source.s_mask
        source.szz             - time varying stress at each of the source
                                positions given by source.s_mask
        source.sxy             - time varying stress at each of the source
                                positions given by source.s_mask
        source.sxz             - time varying stress at each of the source
                                positions given by source.s_mask
        source.syz             - time varying stress at each of the source
                                positions given by source.s_mask
        source.s_mask          - binary matrix specifying the positions of
                                the time varying stress source distributions
        source.s_mode          - optional input to control whether the input
                                stress is injected as a mass source or
                                enforced as a dirichlet boundary condition
                                valid inputs are 'additive' (the default) or
                                'dirichlet'
        source.ux              - time varying particle velocity in the
                                x-direction at each of the source positions
                                given by source.u_mask
        source.uy              - time varying particle velocity in the
                                y-direction at each of the source positions
                                given by source.u_mask
        source.uz              - time varying particle velocity in the
                                z-direction at each of the source positions
                                given by source.u_mask
        source.u_mask          - binary matrix specifying the positions of
                                the time varying particle velocity
                                distribution
        source.u_mode          - optional input to control whether the input
                                velocity is applied as a force source or
                                enforced as a dirichlet boundary condition
                                valid inputs are 'additive' (the default) or
                                'dirichlet'

        sensor.mask*           - binary matrix or a set of Cartesian points
                                where the pressure is recorded at each
                                time-step
        sensor.record          - cell array of the acoustic parameters to
                                record in the form sensor.record = ['p',
                                'u'] valid inputs are:

            'p' (acoustic pressure)
            'p_max' (maximum pressure)
            'p_min' (minimum pressure)
            'p_rms' (RMS pressure)
            'p_final' (final pressure field at all grid points)
            'p_max_all' (maximum pressure at all grid points)
            'p_min_all' (minimum pressure at all grid points)
            'u' (particle velocity)
            'u_max' (maximum particle velocity)
            'u_min' (minimum particle velocity)
            'u_rms' (RMS particle21st January 2014 velocity)
            'u_final' (final particle velocity field at all grid points)
            'u_max_all' (maximum particle velocity at all grid points)
            'u_min_all' (minimum particle velocity at all grid points)
            'u_non_staggered' (particle velocity on non-staggered grid)
            'u_split_field' (particle velocity on non-staggered grid split
                            into compressional and shear components)
            'I' (time varying acoustic intensity)
            'I_avg' (average acoustic intensity)

            NOTE: the acoustic pressure outputs are calculated from the
            normal stress via: p = -(sxx + syy + szz)/3

        sensor.record_start_index
                                - time index at which the sensor should start
                                recording the data specified by
                                sensor.record (default = 1)
        sensor.time_reversal_boundary_data
                                - time varying pressure enforced as a
                                Dirichlet boundary condition over
                                sensor.mask

    Note: For a heterogeneous medium, medium.sound_speed_compression,
    medium.sound_speed_shear, and medium.density must be given in matrix form
    with the same dimensions as kgrid. For a homogeneous medium, these can be
    given as scalar values.

    OPTIONAL INPUTS:
        Optional 'string', value pairs that may be used to modify the default
        computational settings.

        See .html help file for details.

    OUTPUTS:
    If sensor.record is not defined by the user:
        sensor_data            - time varying pressure recorded at the sensor
                                positions given by sensor.mask

    If sensor.record is defined by the user:
        sensor_data.p          - time varying pressure recorded at the
                                sensor positions given by sensor.mask
                                (returned if 'p' is set)
        sensor_data.p_max      - maximum pressure recorded at the sensor
                                positions given by sensor.mask (returned if
                                'p_max' is set)
        sensor_data.p_min      - minimum pressure recorded at the sensor
                                positions given by sensor.mask (returned if
                                'p_min' is set)
        sensor_data.p_rms      - rms of the time varying pressure recorded
                                at the sensor positions given by
                                sensor.mask (returned if 'p_rms' is set)
        sensor_data.p_final    - final pressure field at all grid points
                                within the domain (returned if 'p_final' is
                                set)
        sensor_data.p_max_all  - maximum pressure recorded at all grid points
                                within the domain (returned if 'p_max_all'
                                is set)
        sensor_data.p_min_all  - minimum pressure recorded at all grid points
                                within the domain (returned if 'p_min_all'
                                is set)
        sensor_data.ux         - time varying particle velocity in the
                                x-direction recorded at the sensor positions
                                given by sensor.mask (returned if 'u' is
                                set)
        sensor_data.uy         - time varying particle velocity in the
                                y-direction recorded at the sensor positions
                                given by sensor.mask (returned if 'u' is
                                set)
        sensor_data.uz         - time varying particle velocity in the
                                z-direction recorded at the sensor positions
                                given by sensor.mask (returned if 'u' is
                                set)
        sensor_data.ux_max     - maximum particle velocity in the x-direction
                                recorded at the sensor positions given by
                                sensor.mask (returned if 'u_max' is set)
        sensor_data.uy_max     - maximum particle velocity in the y-direction
                                recorded at the sensor positions given by
                                sensor.mask (returned if 'u_max' is set)
        sensor_data.uz_max     - maximum particle velocity in the z-direction
                                recorded at the sensor positions given by
                                sensor.mask (returned if 'u_max' is set)
        sensor_data.ux_min     - minimum particle velocity in the x-direction
                                recorded at the sensor positions given by
                                sensor.mask (returned if 'u_min' is set)
        sensor_data.uy_min     - minimum particle velocity in the y-direction
                                recorded at the sensor positions given by
                                sensor.mask (returned if 'u_min' is set)
        sensor_data.uz_min     - minimum particle velocity in the z-direction
                                recorded at the sensor positions given by
                                sensor.mask (returned if 'u_min' is set)
        sensor_data.ux_rms     - rms of the time varying particle velocity in
                                the x-direction recorded at the sensor
                                positions given by sensor.mask (returned if
                                'u_rms' is set)
        sensor_data.uy_rms     - rms of the time varying particle velocity in
                                the y-direction recorded at the sensor
                                positions given by sensor.mask (returned if
                                'u_rms' is set)
        sensor_data.uz_rms     - rms of the time varying particle velocity
                                in the z-direction recorded at the sensor
                                positions given by sensor.mask (returned if
                                'u_rms' is set)
        sensor_data.ux_final   - final particle velocity field in the
                                x-direction at all grid points within the
                                domain (returned if 'u_final' is set)
        sensor_data.uy_final   - final particle velocity field in the
                                y-direction at all grid points within the
                                domain (returned if 'u_final' is set)
        sensor_data.uz_final   - final particle velocity field in the
                                z-direction at all grid points within the
                                domain (returned if 'u_final' is set)
        sensor_data.ux_max_all - maximum particle velocity in the x-direction
                                recorded at all grid points within the
                                domain (returned if 'u_max_all' is set)
        sensor_data.uy_max_all - maximum particle velocity in the y-direction
                                recorded at all grid points within the
                                domain (returned if 'u_max_all' is set)
        sensor_data.uz_max_all - maximum particle velocity in the z-direction
                                recorded at all grid points within the
                                domain (returned if 'u_max_all' is set)
        sensor_data.ux_min_all - minimum particle velocity in the x-direction
                                recorded at all grid points within the
                                domain (returned if 'u_min_all' is set)
        sensor_data.uy_min_all - minimum particle velocity in the y-direction
                                recorded at all grid points within the
                                domain (returned if 'u_min_all' is set)
        sensor_data.uz_min_all - minimum particle velocity in the z-direction
                                recorded at all grid points within the
                                domain (returned if 'u_min_all' is set)
        sensor_data.ux_non_staggered
                                - time varying particle velocity in the
                                x-direction recorded at the sensor positions
                                given by sensor.mask after shifting to the
                                non-staggered grid (returned if
                                'u_non_staggered' is set)
        sensor_data.uy_non_staggered
                                - time varying particle velocity in the
                                y-direction recorded at the sensor positions
                                given by sensor.mask after shifting to the
                                non-staggered grid (returned if
                                'u_non_staggered' is set)
        sensor_data.uz_non_staggered
                                - time varying particle velocity in the
                                z-direction recorded at the sensor positions
                                given by sensor.mask after shifting to the
                                non-staggered grid (returned if
                                'u_non_staggered' is set)
        sensor_data.ux_split_p - compressional component of the time varying
                                particle velocity in the x-direction on the
                                non-staggered grid recorded at the sensor
                                positions given by sensor.mask (returned if
                                'u_split_field' is set)
        sensor_data.ux_split_s - shear component of the time varying particle
                                velocity in the x-direction on the
                                non-staggered grid recorded at the sensor
                                positions given by sensor.mask (returned if
                                'u_split_field' is set)
        sensor_data.uy_split_p - compressional component of the time varying
                                particle velocity in the y-direction on the
                                non-staggered grid recorded at the sensor
                                positions given by sensor.mask (returned if
                                'u_split_field' is set)
        sensor_data.uy_split_s - shear component of the time varying particle
                                velocity in the y-direction on the
                                non-staggered grid recorded at the sensor
                                positions given by sensor.mask (returned if
                                'u_split_field' is set)
        sensor_data.uz_split_p - compressional component of the time varying
                                particle velocity in the z-direction on the
                                non-staggered grid recorded at the sensor
                                positions given by sensor.mask (returned if
                                'u_split_field' is set)
        sensor_data.uz_split_s - shear component of the time varying particle
                                velocity in the z-direction on the
                                non-staggered grid recorded at the sensor
                                positions given by sensor.mask (returned if
                                'u_split_field' is set)
        sensor_data.Ix         - time varying acoustic intensity in the
                                x-direction recorded at the sensor positions
                                given by sensor.mask (returned if 'I' is
                                set)
        sensor_data.Iy         - time varying acoustic intensity in the
                                y-direction recorded at the sensor positions
                                given by sensor.mask (returned if 'I' is
                                set)
        sensor_data.Iz         - time varying acoustic intensity in the
                                z-direction recorded at the sensor positions
                                given by sensor.mask (returned if 'I' is
                                set)
        sensor_data.Ix_avg     - average acoustic intensity in the
                                x-direction recorded at the sensor positions
                                given by sensor.mask (returned if 'I_avg' is
                                set)
        sensor_data.Iy_avg     - average acoustic intensity in the
                                y-direction recorded at the sensor positions
                                given by sensor.mask (returned if 'I_avg' is
                                set)
        sensor_data.Iz_avg     - average acoustic intensity in the
                                z-direction recorded at the sensor positions
                                given by sensor.mask (returned if 'I_avg' is
                                set)

    ABOUT:
        author                 - Bradley Treeby & Ben Cox
        date                   - 11th March 2013
        last update            - 13th January 2019

    This function is part of the k-Wave Toolbox (http://www.k-wave.org)
    Copyright (C) 2013-2019 Bradley Treeby and Ben Cox

    See also kspaceFirstOrder3D, kWaveGrid, pstdElastic2D

    This file is part of k-Wave. k-Wave is free software: you can
    redistribute it and/or modify it under the terms of the GNU Lesser
    General Public License as published by the Free Software Foundation,
    either version 3 of the License, or (at your option) any later version.

    k-Wave is distributed in the hope that it will be useful, but WITHOUT ANY
    WARRANTY without even the implied warranty of MERCHANTABILITY or FITNESS
    FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for
    more details.

    You should have received a copy of the GNU Lesser General Public License
    along with k-Wave. If not, see <http://www.gnu.org/licenses/>.
    """

# =========================================================================
# CHECK INPUT STRUCTURES AND OPTIONAL INPUTS
# =========================================================================

    myOrder = 'F'

    # start the timer and store the start time
    timer = TicToc()
    timer.tic()

    # build simulation object with flags and formatted data containers
    k_sim = kWaveSimulation(kgrid=kgrid,
                            source=source,
                            sensor=sensor,
                            medium=medium,
                            simulation_options=simulation_options)

    # run helper script to check inputs
    k_sim.input_checking('pstd_elastic_3d')

    # TODO - if cuboid corners with more than one choise, then is a list,
    sensor_data = k_sim.sensor_data

    options = k_sim.options

    rho0 = np.atleast_1d(k_sim.rho0) # maybe at least 3d?

    m_rho0 : int = np.squeeze(rho0).ndim

    # assign the lame parameters
    mu = medium.sound_speed_shear**2 * medium.density
    lame_lambda = medium.sound_speed_compression**2 * medium.density - 2.0 * mu
    m_mu : int = np.squeeze(mu).ndim

    # assign the viscosity coefficients
    if (options.kelvin_voigt_model):
        eta = 2.0 * rho0 * medium.sound_speed_shear**3 * db2neper(medium.alpha_coeff_shear, 2)
        chi = 2.0 * rho0 * medium.sound_speed_compression**3 * db2neper(medium.alpha_coeff_compression, 2) - 2.0 * eta
        m_eta : int = np.squeeze(eta).ndim


    # =========================================================================
    # CALCULATE MEDIUM PROPERTIES ON STAGGERED GRID
    # =========================================================================

    # calculate the values of the density at the staggered grid points
    # using the arithmetic average [1, 2], where sgx  = (x + dx/2, y),
    # sgy  = (x, y + dy/2) and sgz = (x, y, z + dz/2)
    if (m_rho0 == 3 and (options.use_sg)):
        # rho0 is heterogeneous and staggered grids are used

        points = (np.squeeze(k_sim.kgrid.x_vec), np.squeeze(k_sim.kgrid.y_vec), np.squeeze(k_sim.kgrid.z_vec))

        mg = np.meshgrid(np.squeeze(k_sim.kgrid.x_vec) + k_sim.kgrid.dx / 2.0,
                         np.squeeze(k_sim.kgrid.y_vec),
                         np.squeeze(k_sim.kgrid.z_vec),
                         indexing='ij',)
        interp_points = np.moveaxis(mg, 0, -1)
        rho0_sgx = interpn(points, k_sim.rho0, interp_points,  method='linear', bounds_error=False)

        mg = np.meshgrid(np.squeeze(k_sim.kgrid.x_vec),
                         np.squeeze(k_sim.kgrid.y_vec) + k_sim.kgrid.dy / 2.0,
                         np.squeeze(k_sim.kgrid.z_vec),
                         indexing='ij')
        interp_points = np.moveaxis(mg, 0, -1)
        rho0_sgy = interpn(points, k_sim.rho0, interp_points, method='linear', bounds_error=False)

        mg = np.meshgrid(np.squeeze(k_sim.kgrid.x_vec),
                         np.squeeze(k_sim.kgrid.y_vec),
                         np.squeeze(k_sim.kgrid.z_vec) + k_sim.kgrid.dz / 2.0,
                         indexing='ij')
        interp_points = np.moveaxis(mg, 0, -1)
        rho0_sgz = interpn(points, k_sim.rho0, interp_points, method='linear', bounds_error=False)

        # set values outside of the interpolation range to original values
        rho0_sgx[np.isnan(rho0_sgx)] = rho0[np.isnan(rho0_sgx)]
        rho0_sgy[np.isnan(rho0_sgy)] = rho0[np.isnan(rho0_sgy)]
        rho0_sgz[np.isnan(rho0_sgz)] = rho0[np.isnan(rho0_sgz)]
    else:
        # rho0 is homogeneous or staggered grids are not used
        rho0_sgx = rho0
        rho0_sgy = rho0
        rho0_sgz = rho0

    # elementwise reciprocal of rho0 so it doesn't have to be done each time step
    rho0_sgx_inv = 1.0 / rho0_sgx
    rho0_sgy_inv = 1.0 / rho0_sgy
    rho0_sgz_inv = 1.0 / rho0_sgz

    # clear unused variables if not using them in _saveToDisk
    if not options.save_to_disk:
        del rho0_sgx
        del rho0_sgy
        del rho0_sgz

    # calculate the values of mu at the staggered grid points using the
    # harmonic average [1, 2], where sgxy = (x + dx/2, y + dy/2, z), etc
    if (m_mu == 3 and options.use_sg):
        # interpolation points
        points = (np.squeeze(k_sim.kgrid.x_vec), np.squeeze(k_sim.kgrid.y_vec), np.squeeze(k_sim.kgrid.z_vec))

        # mu is heterogeneous and staggered grids are used
        mg = np.meshgrid(np.squeeze(k_sim.kgrid.x_vec) + k_sim.kgrid.dx / 2.0,
                         np.squeeze(k_sim.kgrid.y_vec) + k_sim.kgrid.dy / 2.0,
                         np.squeeze(k_sim.kgrid.z_vec),
                         indexing='ij')
        interp_points = np.moveaxis(mg, 0, -1)
        with np.errstate(divide='ignore', invalid='ignore'):
            mu_sgxy = 1.0 / interpn(points, 1.0 / mu, interp_points, method='linear', bounds_error=False)

        mg = np.meshgrid(np.squeeze(k_sim.kgrid.x_vec) + k_sim.kgrid.dx / 2.0,
                         np.squeeze(k_sim.kgrid.y_vec),
                         np.squeeze(k_sim.kgrid.z_vec) + k_sim.kgrid.dz / 2.0,
                         indexing='ij')
        interp_points = np.moveaxis(mg, 0, -1)
        with np.errstate(divide='ignore', invalid='ignore'):
            mu_sgxz = 1.0 / interpn(points, 1.0 / mu, interp_points, method='linear', bounds_error=False)

        mg = np.meshgrid(np.squeeze(k_sim.kgrid.x_vec),
                         np.squeeze(k_sim.kgrid.y_vec) + k_sim.kgrid.dy / 2.0,
                         np.squeeze(k_sim.kgrid.z_vec) + k_sim.kgrid.dz / 2.0,
                         indexing='ij')
        interp_points = np.moveaxis(mg, 0, -1)
        with np.errstate(divide='ignore', invalid='ignore'):
            mu_sgyz = 1.0 / interpn(points, 1.0 / mu, interp_points, method='linear', bounds_error=False)

        # set values outside of the interpolation range to original values
        mu_sgxy[np.isnan(mu_sgxy)] = mu[np.isnan(mu_sgxy)]
        mu_sgxz[np.isnan(mu_sgxz)] = mu[np.isnan(mu_sgxz)]
        mu_sgyz[np.isnan(mu_sgyz)] = mu[np.isnan(mu_sgyz)]

    else:
        # mu is homogeneous or staggered grids are not used
        mu_sgxy  = mu
        mu_sgxz  = mu
        mu_sgyz  = mu


    # calculate the values of eta at the staggered grid points using the
    # harmonic average [1, 2], where sgxy = (x + dx/2, y + dy/2, z) etc
    if options.kelvin_voigt_model:
        if m_eta == 3 and options.use_sg:

            points = (np.squeeze(k_sim.kgrid.x_vec), np.squeeze(k_sim.kgrid.y_vec), np.squeeze(k_sim.kgrid.z_vec))

            # eta is heterogeneous and staggered grids are used
            mg = np.meshgrid(np.squeeze(k_sim.kgrid.x_vec) + k_sim.kgrid.dx / 2.0,
                             np.squeeze(k_sim.kgrid.y_vec) + k_sim.kgrid.dy / 2.0,
                             np.squeeze(k_sim.kgrid.z_vec),
                             indexing='ij')
            interp_points = np.moveaxis(mg, 0, -1)
            with np.errstate(divide='ignore', invalid='ignore'):
                eta_sgxy = 1.0 / interpn(points, 1.0 / eta, interp_points, method='linear', bounds_error=False)

            mg = np.meshgrid(np.squeeze(k_sim.kgrid.x_vec) + k_sim.kgrid.dx / 2.0,
                             np.squeeze(k_sim.kgrid.y_vec),
                             np.squeeze(k_sim.kgrid.z_vec) + k_sim.kgrid.dz / 2.0,
                             indexing='ij')
            interp_points = np.moveaxis(mg, 0, -1)
            with np.errstate(divide='ignore', invalid='ignore'):
                eta_sgxz = 1.0 / interpn(points, 1.0 / eta, interp_points, method='linear', bounds_error=False)

            mg = np.meshgrid(np.squeeze(k_sim.kgrid.x_vec),
                             np.squeeze(k_sim.kgrid.y_vec) + k_sim.kgrid.dy / 2.0,
                             np.squeeze(k_sim.kgrid.z_vec) + k_sim.kgrid.dz / 2.0,
                             indexing='ij')
            interp_points = np.moveaxis(mg, 0, -1)
            with np.errstate(divide='ignore', invalid='ignore'):
                eta_sgyz = 1.0 / interpn(points, 1.0 / eta, interp_points, method='linear', bounds_error=False)

            # set values outside of the interpolation range to original values
            eta_sgxy[np.isnan(eta_sgxy)] = eta[np.isnan(eta_sgxy)]
            eta_sgxz[np.isnan(eta_sgxz)] = eta[np.isnan(eta_sgxz)]
            eta_sgyz[np.isnan(eta_sgyz)] = eta[np.isnan(eta_sgyz)]

        else:

            # eta is homogeneous or staggered grids are not used
            eta_sgxy = eta
            eta_sgxz = eta
            eta_sgyz = eta



    # [1] Moczo, P., Kristek, J., Vavry?uk, V., Archuleta, R. J., & Halada, L.
    # (2002). 3D heterogeneous staggered-grid finite-difference modeling of
    # seismic motion with volume harmonic and arithmetic averaging of elastic
    # moduli and densities. Bulletin of the Seismological Society of America,
    # 92(8), 3042-3066.

    # [2] Toyoda, M., Takahashi, D., & Kawai, Y. (2012). Averaged material
    # parameters and boundary conditions for the vibroacoustic
    # finite-difference time-domain method with a nonuniform mesh. Acoustical
    # Science and Technology, 33(4), 273-276.

    # =========================================================================
    # RECORDER
    # =========================================================================

    record = k_sim.record


    # =========================================================================
    # PREPARE DERIVATIVE AND PML OPERATORS
    # =========================================================================

    # get the regular PML operators based on the reference sound speed and PML settings
    Nx, Ny, Nz = k_sim.kgrid.Nx, k_sim.kgrid.Ny, k_sim.kgrid.Nz
    dx, dy, dz = k_sim.kgrid.dx, k_sim.kgrid.dy, k_sim.kgrid.dz
    dt = k_sim.kgrid.dt

    pml_x_alpha, pml_y_alpha, pml_z_alpha  = options.pml_x_alpha, options.pml_y_alpha, options.pml_z_alpha
    pml_x_size, pml_y_size, pml_z_size = options.pml_x_size, options.pml_y_size, options.pml_z_size

    multi_axial_pml_ratio = options.multi_axial_PML_ratio

    c_ref = k_sim.c_ref

    # get the regular PML operators based on the reference sound speed and PML settings
    pml_x     = get_pml(Nx, dx, dt, c_ref, pml_x_size, pml_x_alpha, False, 0)
    pml_x_sgx = get_pml(Nx, dx, dt, c_ref, pml_x_size, pml_x_alpha, options.use_sg, 0)
    pml_y     = get_pml(Ny, dy, dt, c_ref, pml_y_size, pml_y_alpha, False, 1)
    pml_y_sgy = get_pml(Ny, dy, dt, c_ref, pml_y_size, pml_y_alpha, options.use_sg, 1)
    pml_z     = get_pml(Nz, dz, dt, c_ref, pml_z_size, pml_z_alpha, False, 2)
    pml_z_sgz = get_pml(Nz, dz, dt, c_ref, pml_z_size, pml_z_alpha, options.use_sg, 2)

    # get the multi-axial PML operators
    mpml_x     = get_pml(Nx, dx, dt, c_ref, pml_x_size, multi_axial_pml_ratio * pml_x_alpha, False, 0)
    mpml_x_sgx = get_pml(Nx, dx, dt, c_ref, pml_x_size, multi_axial_pml_ratio * pml_x_alpha, options.use_sg, 0)
    mpml_y     = get_pml(Ny, dy, dt, c_ref, pml_y_size, multi_axial_pml_ratio * pml_y_alpha, False, 1)
    mpml_y_sgy = get_pml(Ny, dy, dt, c_ref, pml_y_size, multi_axial_pml_ratio * pml_y_alpha, options.use_sg, 1)
    mpml_z     = get_pml(Nz, dz, dt, c_ref, pml_z_size, multi_axial_pml_ratio * pml_z_alpha, False, 2)
    mpml_z_sgz = get_pml(Nz, dz, dt, c_ref, pml_z_size, multi_axial_pml_ratio * pml_z_alpha, options.use_sg, 2)

    # define the k-space derivative operators, multiply by the staggered
    # grid shift operators, and then re-order using  np.fft.ifftshift (the option
    # options.use_sg exists for debugging)
    if options.use_sg:

        kx_vec = np.squeeze(k_sim.kgrid.k_vec[0])
        ky_vec = np.squeeze(k_sim.kgrid.k_vec[1])
        kz_vec = np.squeeze(k_sim.kgrid.k_vec[2])

        ddx_k_shift_pos =  np.fft.ifftshift(1j * kx_vec * np.exp(1j * kx_vec * kgrid.dx / 2.0))
        ddx_k_shift_neg =  np.fft.ifftshift(1j * kx_vec * np.exp(-1j * kx_vec * kgrid.dx / 2.0))
        ddy_k_shift_pos =  np.fft.ifftshift(1j * ky_vec * np.exp(1j * ky_vec * kgrid.dy / 2.0))
        ddy_k_shift_neg =  np.fft.ifftshift(1j * ky_vec * np.exp(-1j * ky_vec * kgrid.dy / 2.0))
        ddz_k_shift_pos =  np.fft.ifftshift(1j * kz_vec * np.exp(1j * kz_vec * kgrid.dz / 2.0))
        ddz_k_shift_neg =  np.fft.ifftshift(1j * kz_vec * np.exp(-1j * kz_vec * kgrid.dz / 2.0))
    else:
        ddx_k_shift_pos =  np.fft.ifftshift(1j * kx_vec)
        ddx_k_shift_neg =  np.fft.ifftshift(1j * kx_vec)
        ddy_k_shift_pos =  np.fft.ifftshift(1j * ky_vec)
        ddy_k_shift_neg =  np.fft.ifftshift(1j * ky_vec)
        ddz_k_shift_pos =  np.fft.ifftshift(1j * kz_vec)
        ddz_k_shift_neg =  np.fft.ifftshift(1j * kz_vec)

    # force the derivative and shift operators to be in the correct direction
    # for use with broadcasting
    ddx_k_shift_pos = np.expand_dims(np.expand_dims(np.squeeze(ddx_k_shift_pos), axis=-1), axis=-1)
    ddx_k_shift_neg = np.expand_dims(np.expand_dims(np.squeeze(ddx_k_shift_neg), axis=-1), axis=-1)
    ddy_k_shift_pos = np.expand_dims(np.expand_dims(np.squeeze(ddy_k_shift_pos), axis=0), axis=-1)
    ddy_k_shift_neg = np.expand_dims(np.expand_dims(np.squeeze(ddy_k_shift_neg), axis=0), axis=-1)
    ddz_k_shift_pos = np.expand_dims(np.expand_dims(np.squeeze(ddz_k_shift_pos), axis=0), axis=0)
    ddz_k_shift_neg = np.expand_dims(np.expand_dims(np.squeeze(ddz_k_shift_neg), axis=0), axis=0)

    ddx_k_shift_pos = np.reshape(ddx_k_shift_pos, ddx_k_shift_pos.shape, order='F')
    ddx_k_shift_neg = np.reshape(ddx_k_shift_neg, ddx_k_shift_neg.shape, order='F')
    ddy_k_shift_pos = np.reshape(ddy_k_shift_pos, ddy_k_shift_pos.shape, order='F')
    ddy_k_shift_neg = np.reshape(ddy_k_shift_neg, ddy_k_shift_neg.shape, order='F')
    ddz_k_shift_pos = np.reshape(ddz_k_shift_pos, ddz_k_shift_pos.shape, order='F')
    ddz_k_shift_neg = np.reshape(ddz_k_shift_neg, ddz_k_shift_neg.shape, order='F')

    pml_x = np.transpose(pml_x)
    pml_x_sgx = np.transpose(pml_x_sgx)
    mpml_x = np.transpose(mpml_x)
    mpml_x_sgx = np.transpose(mpml_x_sgx)

    pml_x = np.expand_dims(pml_x, axis=-1)
    pml_x_sgx = np.expand_dims(pml_x_sgx, axis=-1)
    mpml_x = np.expand_dims(mpml_x, axis=-1)
    mpml_x_sgx = np.expand_dims(mpml_x_sgx, axis=-1)

    pml_y = np.expand_dims(pml_y, axis=0)
    pml_y_sgy = np.expand_dims(pml_y_sgy, axis=0)
    mpml_y = np.expand_dims(mpml_y, axis=0)
    mpml_y_sgy = np.expand_dims(mpml_y_sgy, axis=0)

    pml_z = np.expand_dims(pml_z, axis=0)
    pml_z_sgz = np.expand_dims(pml_z_sgz, axis=0)
    mpml_z = np.expand_dims(mpml_z, axis=0)
    mpml_z_sgz = np.expand_dims(mpml_z_sgz, axis=0)

    # =========================================================================
    # DATA CASTING
    # =========================================================================

    # run subscript to cast the remaining loop variables to the data type
    # specified by data_cast
    if not (options.data_cast == 'off'):
        myType = np.single
    else:
        myType = np.double

    grid_shape = (Nx, Ny, Nz)

    # preallocate the loop variables using the castZeros anonymous function
    # (this creates a matrix of zeros in the data type specified by data_cast)
    ux_split_x  = np.zeros(grid_shape, myType, order=myOrder)
    ux_split_y  = np.zeros(grid_shape, myType, order=myOrder)
    ux_split_z  = np.zeros(grid_shape, myType, order=myOrder)
    ux_sgx      = np.zeros(grid_shape, myType, order=myOrder)  # **
    uy_split_x  = np.zeros(grid_shape, myType, order=myOrder)
    uy_split_y  = np.zeros(grid_shape, myType, order=myOrder)
    uy_split_z  = np.zeros(grid_shape, myType, order=myOrder)
    uy_sgy      = np.zeros(grid_shape, myType, order=myOrder)  # **
    uz_split_x  = np.zeros(grid_shape, myType, order=myOrder)
    uz_split_y  = np.zeros(grid_shape, myType, order=myOrder)
    uz_split_z  = np.zeros(grid_shape, myType, order=myOrder)
    uz_sgz      = np.zeros(grid_shape, myType, order=myOrder)  # **

    sxx_split_x = np.zeros(grid_shape, myType, order=myOrder)
    sxx_split_y = np.zeros(grid_shape, myType, order=myOrder)
    sxx_split_z = np.zeros(grid_shape, myType, order=myOrder)
    syy_split_x = np.zeros(grid_shape, myType, order=myOrder)
    syy_split_y = np.zeros(grid_shape, myType, order=myOrder)
    syy_split_z = np.zeros(grid_shape, myType, order=myOrder)
    szz_split_x = np.zeros(grid_shape, myType, order=myOrder)
    szz_split_y = np.zeros(grid_shape, myType, order=myOrder)
    szz_split_z = np.zeros(grid_shape, myType, order=myOrder)
    sxy_split_x = np.zeros(grid_shape, myType, order=myOrder)
    sxy_split_y = np.zeros(grid_shape, myType, order=myOrder)
    sxz_split_x = np.zeros(grid_shape, myType, order=myOrder)
    sxz_split_z = np.zeros(grid_shape, myType, order=myOrder)
    syz_split_y = np.zeros(grid_shape, myType, order=myOrder)
    syz_split_z = np.zeros(grid_shape, myType, order=myOrder)

    duxdx       = np.zeros(grid_shape, myType, order=myOrder)  # **
    duxdy       = np.zeros(grid_shape, myType, order=myOrder)  # **
    duxdz       = np.zeros(grid_shape, myType, order=myOrder)  # **
    duydx       = np.zeros(grid_shape, myType, order=myOrder)  # **
    duydy       = np.zeros(grid_shape, myType, order=myOrder)  # **
    duydz       = np.zeros(grid_shape, myType, order=myOrder)  # **
    duzdx       = np.zeros(grid_shape, myType, order=myOrder)  # **
    duzdy       = np.zeros(grid_shape, myType, order=myOrder)  # **
    duzdz       = np.zeros(grid_shape, myType, order=myOrder)  # **

    dsxxdx      = np.zeros(grid_shape, myType, order=myOrder)  # **
    dsyydy      = np.zeros(grid_shape, myType, order=myOrder)  # **
    dszzdz      = np.zeros(grid_shape, myType, order=myOrder)  # **
    dsxydx      = np.zeros(grid_shape, myType, order=myOrder)  # **
    dsxydy      = np.zeros(grid_shape, myType, order=myOrder)  # **
    dsxzdx      = np.zeros(grid_shape, myType, order=myOrder)  # **
    dsxzdz      = np.zeros(grid_shape, myType, order=myOrder)  # **
    dsyzdy      = np.zeros(grid_shape, myType, order=myOrder)  # **
    dsyzdz      = np.zeros(grid_shape, myType, order=myOrder)  # **

    p           = np.zeros(grid_shape, myType, order=myOrder)  # **

    if options.kelvin_voigt_model:
        dduxdxdt       = np.zeros(grid_shape, myType, order=myOrder)  # **
        dduxdydt       = np.zeros(grid_shape, myType, order=myOrder)  # **
        dduxdzdt       = np.zeros(grid_shape, myType, order=myOrder)  # **
        dduydxdt       = np.zeros(grid_shape, myType, order=myOrder)  # **
        dduydydt       = np.zeros(grid_shape, myType, order=myOrder)  # **
        dduydzdt       = np.zeros(grid_shape, myType, order=myOrder)  # **
        dduzdxdt       = np.zeros(grid_shape, myType, order=myOrder)  # **
        dduzdydt       = np.zeros(grid_shape, myType, order=myOrder)  # **
        dduzdzdt       = np.zeros(grid_shape, myType, order=myOrder)  # **

    # to save memory, the variables noted with a ** do not neccesarily need to
    # be np.explicitly stored (they are not needed for update steps). Instead they
    # could be replaced with a small number of temporary variables that are
    # reused several times during the time loop.


    # =========================================================================
    # CREATE INDEX VARIABLES
    # =========================================================================

    # setup the time index variable
    if not options.time_rev:
        index_start: int = 0
        index_step: int = 1
        index_end: int = k_sim.kgrid.Nt
    else:
        # throw error for unsupported feature
        raise TypeError('Time reversal using sensor.time_reversal_boundary_data is not currently supported.')


    # =========================================================================
    # PREPARE VISUALISATIONS
    # =========================================================================

    # # pre-compute suitable axes scaling factor
    # if (options.plot_layout or options.plot_sim):
    #     x_sc, scale, prefix = scaleSI(np.max([kgrid.x_vec kgrid.y_vec kgrid.z_vec])) ##ok<ASGLU>


    # # throw error for currently unsupported plot layout feature
    # if options.plot_layout:
    #     raise TypeError('"PlotLayout" input is not currently supported.')


    # # initialise the figure used for animation if 'PlotSim' is set to 'True'
    # if options.plot_sim:
    #     kspaceFirstOrder_initialiseFigureWindow

    # # initialise movie parameters if 'RecordMovie' is set to 'True'
    # if options.record_movie:
    #     kspaceFirstOrder_initialiseMovieParameters


    # =========================================================================
    # ENSURE PYTHON INDEXING
    # =========================================================================

    # These should be zero indexed
    if hasattr(k_sim, 's_source_pos_index'):
        if k_sim.s_source_pos_index is not None:
        # if k_sim.s_source_pos_index.ndim != 0:
            k_sim.s_source_pos_index = np.squeeze(np.asarray(k_sim.s_source_pos_index)) - int(1)

    if hasattr(k_sim, 'u_source_pos_index'):
        if k_sim.u_source_pos_index is not None:
        # if k_sim.u_source_pos_index.ndim != 0:
          k_sim.u_source_pos_index = np.squeeze(k_sim.u_source_pos_index) - int(1)

    if hasattr(k_sim, 'p_source_pos_index'):
        if k_sim.p_source_pos_index is not None:
        # if k_sim.p_source_pos_index.ndim != 0:
            k_sim.p_source_pos_index = np.squeeze(k_sim.p_source_pos_index) - int(1)

    if hasattr(k_sim, 's_source_sig_index'):
        if k_sim.s_source_sig_index is not None:
            k_sim.s_source_sig_index = np.squeeze(k_sim.s_source_sig_index) - int(1)

    if hasattr(k_sim, 'u_source_sig_index') and k_sim.u_source_sig_index is not None:
        k_sim.u_source_sig_index = np.squeeze(k_sim.u_source_sig_index) - int(1)

    if hasattr(k_sim, 'p_source_sig_index') and k_sim.p_source_sig_index is not None:
        k_sim.p_source_sig_index = np.squeeze(k_sim.p_source_sig_index) - int(1)

    if hasattr(k_sim, 'sensor_mask_index') and k_sim.sensor_mask_index is not None:
        k_sim.sensor_mask_index = np.squeeze(k_sim.sensor_mask_index) - int(1)

    # These should be zero indexed. Note the x2, y2 and z2 indices do not need to be shifted
    if hasattr(record, 'x1_inside') and record.x1_inside is not None:
      if (record.x1_inside == 0):
          print("GAH")
      else:
          record.x1_inside = int(record.x1_inside - 1)
    if hasattr(record, 'y1_inside') and record.y1_inside is not None:
        record.y1_inside = int(record.y1_inside - 1)
    if hasattr(record, 'z1_inside') and record.z1_inside is not None:
        record.z1_inside = int(record.z1_inside - 1)

    # =========================================================================
    # checking
    # =========================================================================

    checking: bool = False
    import scipy.io as sio
    filename = "C:/Users/dsinden/dev/octave/k-Wave/testing/unit/3DoneStep_split.mat"
    verbose: bool = True
    load_index: int = 0
    if checking:
        mat_contents = sio.loadmat(filename)
        tol: float = 10E-12

        if verbose:
            print(sorted(mat_contents.keys()))

        # mat_mu_sgxy = mat_contents['mu_sgxy']
        # mat_mu_sgxz = mat_contents['mu_sgxz']
        # mat_mu_sgyz = mat_contents['mu_sgyz']


        mat_dsxxdx = mat_contents['dsxxdx']
        mat_dsyydy = mat_contents['dsyydy']
        mat_dsxydx = mat_contents['dsxydx']
        mat_dsxydy = mat_contents['dsxydy']

        mat_sxx = mat_contents['sxx']
        mat_syy = mat_contents['syy']
        mat_szz = mat_contents['szz']

        mat_duxdx = mat_contents['duxdx']
        mat_duxdy = mat_contents['duxdy']
        mat_duydx = mat_contents['duydx']
        mat_duydy = mat_contents['duydy']

        mat_ux_sgx = mat_contents['ux_sgx']
        mat_ux_split_x = mat_contents['ux_split_x']
        mat_ux_split_y = mat_contents['ux_split_y']
        mat_uy_sgy = mat_contents['uy_sgy']
        mat_uy_split_x = mat_contents['uy_split_x']
        mat_uy_split_y = mat_contents['uy_split_y']

        mat_sxx_split_x = mat_contents['sxx_split_x']
        mat_sxx_split_y = mat_contents['sxx_split_y']
        mat_syy_split_x = mat_contents['syy_split_x']
        mat_syy_split_y = mat_contents['syy_split_y']
        mat_sxy_split_x = mat_contents['sxy_split_x']
        mat_sxy_split_y = mat_contents['sxy_split_y']

    # =========================================================================
    # LOOP THROUGH TIME STEPS
    # =========================================================================

    # update command line status
    print('\tprecomputation completed in', scale_time(TicToc.toc()))
    print('\tstarting time loop ...')

    # if k_sim.source_ux is not False:
    #     print(k_sim.source.ux[0:1, 0:1])

    # index_end = 6

    # start time loop
    for t_index in tqdm(np.arange(index_start, index_end, index_step, dtype=int)):

        # compute the gradients of the stress tensor (these variables do not
        # necessaily need to be stored, they could be computed as needed)
        dsxxdx = np.real(np.fft.ifft(np.multiply(ddx_k_shift_pos, np.fft.fft(sxx_split_x + sxx_split_y + sxx_split_z, axis=0), order='F'), axis=0)) #
        dsyydy = np.real(np.fft.ifft(np.multiply(ddy_k_shift_pos, np.fft.fft(syy_split_x + syy_split_y + syy_split_z, axis=1), order='F'), axis=1)) #
        dszzdz = np.real(np.fft.ifft(np.multiply(ddz_k_shift_pos, np.fft.fft(szz_split_x + szz_split_y + szz_split_z, axis=2), order='F'), axis=2)) #

        dsxydx = np.real(np.fft.ifft(np.multiply(ddx_k_shift_neg, np.fft.fft(sxy_split_x + sxy_split_y, axis=0), order='F'), axis=0))
        dsxydy = np.real(np.fft.ifft(np.multiply(ddy_k_shift_neg, np.fft.fft(sxy_split_x + sxy_split_y, axis=1), order='F'), axis=1))

        dsxzdx = np.real(np.fft.ifft(np.multiply(ddx_k_shift_neg, np.fft.fft(sxz_split_x + sxz_split_z, axis=0), order='F'), axis=0))
        dsxzdz = np.real(np.fft.ifft(np.multiply(ddz_k_shift_neg, np.fft.fft(sxz_split_x + sxz_split_z, axis=2), order='F'), axis=2))

        dsyzdy = np.real(np.fft.ifft(np.multiply(ddy_k_shift_neg, np.fft.fft(syz_split_y + syz_split_z, axis=1), order='F'), axis=1))
        dsyzdz = np.real(np.fft.ifft(np.multiply(ddz_k_shift_neg, np.fft.fft(syz_split_y + syz_split_z, axis=2), order='F'), axis=2))

        if checking:
            if (t_index == load_index):
                diff0 = np.max(np.abs(dsxxdx - mat_dsxxdx))
                if (diff0 > tol):
                    print("\tdsxxdx is incorrect:" + diff0)
                diff0 = np.max(np.abs(dsyydy - mat_dsyydy))
                if (diff0 > tol):
                    print("\tdsyydy is incorrect:" + diff0)
                diff0 = np.max(np.abs(dsxydy - mat_dsxydy))
                if (diff0 > tol):
                    print("\tdsxydy is incorrect:" + diff0)
                diff0 = np.max(np.abs(dsxydx - mat_dsxydx))
                if (diff0 > tol):
                    print("\tdsxydx is incorrect:" + diff0)


        # calculate the split-field components of ux_sgx, uy_sgy, and uz_sgz at
        # the next time step using the components of the stress at the current
        # time step

        # ux_split_x = bsxfun(times, mpml_z,
        #                     bsxfun(times, mpml_y,e
        #                            bsxfun(times, pml_x_sgx,d
        #                                   bsxfun(times, mpml_z,c
        #                                          bsxfun(times, mpml_y, b
        #                                                 bsxfun(times, pml_x_sgx, ux_split_x))) + kgrid.dt * rho0_sgx_inv * dsxxdx))) a,c
        a = np.multiply(pml_x_sgx, ux_split_x, order='F')
        b = np.multiply(mpml_y, a, order='F')
        c = np.multiply(mpml_z, b, order='F')
        c = c + kgrid.dt * np.multiply(rho0_sgx_inv, dsxxdx, order='F')
        d = np.multiply(pml_x_sgx, c, order='F')
        e = np.multiply(mpml_y, d, order='F')
        ux_split_x = np.multiply(mpml_z, e, order='F')

        # ux_split_y = bsxfun(@times, mpml_x_sgx, bsxfun(@times, mpml_z,     bsxfun(@times, pml_y, \
        #             bsxfun(@times, mpml_x_sgx, bsxfun(@times, mpml_z,     bsxfun(@times, pml_y, ux_split_y))) \
        #             + kgrid.dt * rho0_sgx_inv * dsxydy)))
        a = np.multiply(pml_y, ux_split_y, order='F')
        b = np.multiply(mpml_z, a, order='F')
        c = np.multiply(mpml_x_sgx, b, order='F')
        c = c + kgrid.dt * np.multiply(rho0_sgx_inv, dsxydy, order='F')
        d = np.multiply(pml_y, c, order='F')
        e = np.multiply(mpml_z, d, order='F')
        ux_split_y = np.multiply(mpml_x_sgx, e, order='F')

        # ux_split_z = bsxfun(@times, mpml_y,     bsxfun(@times, mpml_x_sgx, bsxfun(@times, pml_z, \
        #             bsxfun(@times, mpml_y,     bsxfun(@times, mpml_x_sgx, bsxfun(@times, pml_z, ux_split_z))) \
        #             + kgrid.dt * rho0_sgx_inv * dsxzdz)))
        a = np.multiply(pml_z, ux_split_z, order='F')
        b = np.multiply(mpml_x_sgx, a, order='F')
        c = np.multiply(mpml_y, b, order='F')
        c = c + kgrid.dt * np.multiply(rho0_sgx_inv, dsxzdz, order='F')
        d = np.multiply(pml_z, c, order='F')
        e = np.multiply(mpml_x_sgx, d, order='F')
        ux_split_z = np.multiply(mpml_y, e, order='F')

        # uy_split_x = bsxfun(@times, mpml_z,     bsxfun(@times, mpml_y_sgy, bsxfun(@times, pml_x, \
        #             bsxfun(@times, mpml_z,     bsxfun(@times, mpml_y_sgy, bsxfun(@times, pml_x, uy_split_x))) \
        #             + kgrid.dt * rho0_sgy_inv * dsxydx)))
        a = np.multiply(pml_x, uy_split_x, order='F')
        b = np.multiply(mpml_y_sgy, a, order='F')
        c = np.multiply(mpml_z, b, order='F')
        c = c + kgrid.dt * np.multiply(rho0_sgy_inv, dsxydx, order='F')
        d = np.multiply(pml_x, c, order='F')
        e = np.multiply(mpml_y_sgy, d, order='F')
        uy_split_x = np.multiply(mpml_z, e, order='F')

        # uy_split_y = bsxfun(@times, mpml_x,     bsxfun(@times, mpml_z, bsxfun(@times, pml_y_sgy, \
        #             bsxfun(@times, mpml_x,     bsxfun(@times, mpml_z, bsxfun(@times, pml_y_sgy, uy_split_y))) \
        #             + kgrid.dt * rho0_sgy_inv * dsyydy)))
        a = np.multiply(pml_y_sgy, uy_split_y, order='F')
        b = np.multiply(mpml_z, a, order='F')
        c = np.multiply(mpml_x, b, order='F')
        c = c + kgrid.dt * np.multiply(rho0_sgy_inv, dsyydy, order='F')
        d = np.multiply(pml_y_sgy, c, order='F')
        e = np.multiply(mpml_z, d, order='F')
        uy_split_y = np.multiply(mpml_x, e, order='F')

        # uy_split_z = bsxfun(@times, mpml_y_sgy,
        #               bsxfun(@times, mpml_x,
        #                 bsxfun(@times, pml_z, \
        #                   bsxfun(@times, mpml_y_sgy,
        #                     bsxfun(@times, mpml_x,
        #                       bsxfun(@times, pml_z, uy_split_z))) \
        #                           + kgrid.dt * rho0_sgy_inv * dsyzdz)))
        a = np.multiply(pml_z, uy_split_z, order='F')
        b = np.multiply(mpml_x, a, order='F')
        c = np.multiply(mpml_y_sgy, b, order='F')
        c = c + kgrid.dt * np.multiply(rho0_sgy_inv, dsyzdz, order='F')
        d = np.multiply(pml_z, c, order='F')
        e = np.multiply(mpml_x, d, order='F')
        uy_split_z = np.multiply(mpml_y_sgy, e, order='F')

        # uz_split_x = bsxfun(@times, mpml_z_sgz, bsxfun(@times, mpml_y, bsxfun(@times, pml_x, \
        #             bsxfun(@times, mpml_z_sgz, bsxfun(@times, mpml_y, bsxfun(@times, pml_x, uz_split_x))) \
        #             + kgrid.dt * rho0_sgz_inv * dsxzdx)))
        a = np.multiply(pml_x, uz_split_x, order='F')
        b = np.multiply(mpml_y, a, order='F')
        c = np.multiply(mpml_z_sgz, b, order='F')
        c = c + kgrid.dt * np.multiply(rho0_sgz_inv, dsxzdx, order='F')
        d = np.multiply(pml_x, c, order='F')
        e = np.multiply(mpml_y, d, order='F')
        uz_split_x = np.multiply(mpml_z_sgz, e, order='F')

        # uz_split_y = bsxfun(@times, mpml_x,     bsxfun(@times, mpml_z_sgz, bsxfun(@times, pml_y, \
        #             bsxfun(@times, mpml_x,     bsxfun(@times, mpml_z_sgz, bsxfun(@times, pml_y, uz_split_y))) \
        #             + kgrid.dt * rho0_sgz_inv * dsyzdy)))
        a = np.multiply(pml_y, uz_split_y, order='F')
        b = np.multiply(mpml_z_sgz, a, order='F')
        c = np.multiply(mpml_x, b, order='F')
        c = c + kgrid.dt * np.multiply(rho0_sgz_inv, dsyzdy, order='F')
        d = np.multiply(pml_y, c, order='F')
        e = np.multiply(mpml_z_sgz, d, order='F')
        uz_split_y = np.multiply(mpml_x, e, order='F')

        # uz_split_z = bsxfun(@times, mpml_y,     bsxfun(@times, mpml_x,     bsxfun(@times, pml_z_sgz, ...
        #             bsxfun(@times, mpml_y,     bsxfun(@times, mpml_x,     bsxfun(@times, pml_z_sgz, uz_split_z))) ...
        #             + dt .* rho0_sgz_inv .* dszzdz)));
        # uz_split_z = bsxfun(@times, mpml_y, bsxfun(@times, mpml_x, bsxfun(@times, pml_z_sgz, \
        #             bsxfun(@times, mpml_y, bsxfun(@times, mpml_x, bsxfun(@times, pml_z_sgz, uz_split_z))) \
        #             + kgrid.dt * rho0_sgz_inv * dszzdz)))
        a = np.multiply(pml_z_sgz, uz_split_z, order='F')
        b = np.multiply(mpml_x, a, order='F')
        c = np.multiply(mpml_y, b, order='F')
        c = c + kgrid.dt * np.multiply(rho0_sgz_inv, dszzdz, order='F')
        d = np.multiply(pml_z_sgz, c, order='F')
        e = np.multiply(mpml_x, d, order='F')
        uz_split_z = np.multiply(mpml_y, e, order='F')
        # uz_split_z = mpml_y * mpml_x * pml_z_sgz * (mpml_y * mpml_x * pml_z_sgz * uz_split_z + kgrid.dt * rho0_sgz_inv * dszzdz)

        if checking:
            if (t_index == load_index):

                # print(k_sim.u_source_sig_index)
                # print(k_sim.source.ux[k_sim.u_source_sig_index, t_index])

                diff0 = np.max(np.abs(ux_split_x - mat_ux_split_x))
                if (diff0 > tol):
                    print("\tux_split_x is incorrect:" + diff0)
                diff0 = np.max(np.abs(ux_split_y - mat_ux_split_y))
                if (diff0 > tol):
                    print("\tux_split_y is incorrect:" + diff0)
                diff0 = np.max(np.abs(uy_split_x - mat_uy_split_x))
                if (diff0 > tol):
                    print("\tuy_split_x is incorrect:" + diff0)

        # if (t_index == load_index):

        #     print(k_sim.u_source_sig_index)
        #     print(k_sim.source.ux[k_sim.u_source_sig_index, t_index])


        # add in the velocity source terms
        if k_sim.source_ux is not False and k_sim.source_ux > t_index:
            if (source.u_mode == 'dirichlet'):
                # enforce the source values as a dirichlet boundary condition
                ux_split_x[np.unravel_index(k_sim.u_source_pos_index, ux_split_x.shape, order='F')] = np.squeeze(k_sim.source.ux[k_sim.u_source_sig_index, t_index])
            else:
                # add the source values to the existing field values
                ux_split_x[np.unravel_index(k_sim.u_source_pos_index, ux_split_x.shape, order='F')] = \
                  ux_split_x[np.unravel_index(k_sim.u_source_pos_index, ux_split_x.shape, order='F')] + \
                  np.squeeze(k_sim.source.ux[k_sim.u_source_sig_index, t_index])

        if k_sim.source_uy is not False and k_sim.source_uy > t_index:
            if (source.u_mode == 'dirichlet'):
                # enforce the source values as a dirichlet boundary condition
                uy_split_y[np.unravel_index(k_sim.u_source_pos_index, uy_split_y.shape, order='F')] = np.squeeze(k_sim.source.uy[k_sim.u_source_sig_index, t_index])
            else:
                # add the source values to the existing field values
                uy_split_y[np.unravel_index(k_sim.u_source_pos_index, uy_split_y.shape, order='F')] = \
                  uy_split_y[np.unravel_index(k_sim.u_source_pos_index, uy_split_y.shape, order='F')] + \
                  np.squeeze(k_sim.source.uy[k_sim.u_source_sig_index, t_index])

        if k_sim.source_uz is not False and k_sim.source_uz > t_index:
            if (source.u_mode == 'dirichlet'):
                # enforce the source values as a dirichlet boundary condition
                uz_split_z[np.unravel_index(k_sim.u_source_pos_index, uz_split_z.shape, order='F')] = np.squeeze(k_sim.source.uz[k_sim.u_source_sig_index, t_index])
            else:
                # add the source values to the existing field values
                uz_split_z[np.unravel_index(k_sim.u_source_pos_index, uz_split_z.shape, order='F')] = \
                  uz_split_z[np.unravel_index(k_sim.u_source_pos_index, uz_split_z.shape, order='F')] + \
                  np.squeeze(k_sim.source.uz[k_sim.u_source_sig_index, t_index])


        # combine split field components (these variables do not necessarily
        # need to be stored, they could be computed when needed)
        ux_sgx = ux_split_x + ux_split_y + ux_split_z
        uy_sgy = uy_split_x + uy_split_y + uy_split_z
        uz_sgz = uz_split_x + uz_split_y + uz_split_z

        # calculate the velocity gradients (these variables do not necessarily
        # need to be stored, they could be computed when needed)
        duxdx = np.real(np.fft.ifft(np.multiply(ddx_k_shift_neg, np.fft.fft(ux_sgx, axis=0), order='F'), axis=0)) #
        duxdy = np.real(np.fft.ifft(np.multiply(ddy_k_shift_pos, np.fft.fft(ux_sgx, axis=1), order='F'), axis=1)) #
        duxdz = np.real(np.fft.ifft(np.multiply(ddz_k_shift_pos, np.fft.fft(ux_sgx, axis=2), order='F'), axis=2))

        # changed here! ux_sgy -> uy_sgy
        duydx = np.real(np.fft.ifft(np.multiply(ddx_k_shift_pos, np.fft.fft(uy_sgy, axis=0), order='F'), axis=0))
        duydy = np.real(np.fft.ifft(np.multiply(ddy_k_shift_neg, np.fft.fft(uy_sgy, axis=1), order='F'), axis=1)) #
        duydz = np.real(np.fft.ifft(np.multiply(ddz_k_shift_pos, np.fft.fft(uy_sgy, axis=2), order='F'), axis=2)) #

        # changed here! ux_sgz -> uz_sgz
        duzdx = np.real(np.fft.ifft(np.multiply(ddx_k_shift_pos, np.fft.fft(uz_sgz, axis=0), order='F'), axis=0))
        duzdy = np.real(np.fft.ifft(np.multiply(ddy_k_shift_pos, np.fft.fft(uz_sgz, axis=1), order='F'), axis=1))
        duzdz = np.real(np.fft.ifft(np.multiply(ddz_k_shift_neg, np.fft.fft(uz_sgz, axis=2), order='F'), axis=2))

        if checking:
            if (t_index == load_index):
                diff0 = np.max(np.abs(duxdx - mat_duxdx))
                if (diff0 > tol):
                    print("\tduxdx is incorrect:" + diff0)
                diff0 = np.max(np.abs(duxdy - mat_duxdy))
                if (diff0 > tol):
                    print("\tduxdy is incorrect:" + diff0)
                diff0 = np.max(np.abs(duydy - mat_duydy))
                if (diff0 > tol):
                    print("\tduydy is incorrect:" + diff0)
                diff0 = np.max(np.abs(duydx - mat_duydx))
                if (diff0 > tol):
                    print("\tduydx is incorrect:" + diff0)

        if options.kelvin_voigt_model:

            # compute additional gradient terms needed for the Kelvin-Voigt model
            temp = np.multiply((dsxxdx + dsxydy + dsxzdz), rho0_sgx_inv, order='F')
            dduxdxdt = np.real(np.fft.ifft(np.multiply(ddx_k_shift_neg, np.fft.fft(temp, axis=0), order='F'), axis=0))
            dduxdydt = np.real(np.fft.ifft(np.multiply(ddy_k_shift_pos, np.fft.fft(temp, axis=1), order='F'), axis=1))
            dduxdzdt = np.real(np.fft.ifft(np.multiply(ddz_k_shift_pos, np.fft.fft(temp, axis=2), order='F'), axis=2))

            temp = np.multiply((dsxydx + dsyydy + dsyzdz), rho0_sgy_inv, order='F')
            dduydxdt = np.real(np.fft.ifft(np.multiply(ddx_k_shift_pos, np.fft.fft(temp, axis=0), order='F'), axis=0))
            dduydydt = np.real(np.fft.ifft(np.multiply(ddy_k_shift_neg, np.fft.fft(temp, axis=1), order='F'), axis=1))
            dduydzdt = np.real(np.fft.ifft(np.multiply(ddz_k_shift_pos, np.fft.fft(temp, axis=2), order='F'), axis=2))

            temp = np.multiply((dsxzdx + dsyzdy + dszzdz), rho0_sgz_inv, order='F')
            dduzdxdt = np.real(np.fft.ifft(np.multiply(ddx_k_shift_pos, np.fft.fft(temp, axis=0), order='F'), axis=0))
            dduzdydt = np.real(np.fft.ifft(np.multiply(ddy_k_shift_pos, np.fft.fft(temp, axis=1), order='F'), axis=1))
            dduzdzdt = np.real(np.fft.ifft(np.multiply(ddz_k_shift_neg, np.fft.fft(temp, axis=2), order='F'), axis=2))

            # update the normal shear components of the stress tensor using a
            # Kelvin-Voigt model with a split-field multi-axial pml

            #  sxx_split_x = bsxfun(@times, mpml_z,
            #                  bsxfun(@times, mpml_y,
            #                    bsxfun(@times, pml_x, \
            #                      bsxfun(@times, mpml_z,
            #                        bsxfun(@times, mpml_y,
            #                          bsxfun(@times, pml_x, sxx_split_x) ) ) \
            #                            + kgrid.dt * (2.0 * mu + lame_lambda) * duxdx + kgrid.dt * (2.0 * eta + chi) * dduxdxdt)))
            sxx_split_x = np.multiply(mpml_z,
                            np.multiply(mpml_y,
                              np.multiply(pml_x,
                                np.multiply(mpml_z,
                                  np.multiply(mpml_y,
                                    np.multiply(pml_x, sxx_split_x, order='F'), order='F'), order='F') + \
                                                 kgrid.dt * np.multiply(2.0 * mu + lame_lambda, duxdx, order='F') + \
                                                 kgrid.dt * np.multiply(2.0 * eta + chi, dduxdxdt, order='F'), order='F'), order='F'), order='F')

            # sxx_split_x = mpml_z * (mpml_y * (pml_x * (mpml_z * mpml_y * np.multiply(pml_x, sxx_split_x) + \
            #                                           kgrid.dt * np.multiply(2.0 * mu + lame_lambda, duxdx, order='F') + \
            #                                           kgrid.dt * np.multiply(2.0 * eta + chi, dduxdxdt, order='F') )))


            # sxx_split_y = bsxfun(@times, mpml_x, bsxfun(@times, mpml_z, bsxfun(@times, pml_y, \
            #             bsxfun(@times, mpml_x, bsxfun(@times, mpml_z, bsxfun(@times, pml_y, sxx_split_y))) \
            #             + kgrid.dt * lame_lambda * duydy \
            #             + kgrid.dt * chi * dduydydt)))
            # temp0 = np.copy(sxx_split_y)
            # temp0 = mpml_x * (mpml_z * (pml_y * (mpml_x * (mpml_z * (pml_y * temp0)) +
            #                         kgrid.dt * np.multiply(lame_lambda, duydy, order='F') + kgrid.dt * np.multiply(chi, dduydydt, order='F') )))
            # temp1 = np.copy(sxx_split_y)
            # temp1 = mpml_x * (mpml_z * (pml_y * (mpml_x * (mpml_z * (pml_y * temp1)) +
            #                         kgrid.dt * lame_lambda * duydy + kgrid.dt * chi * dduydydt)))
            # temp2 = np.copy(sxx_split_y)

            a = np.multiply(pml_y, sxx_split_y, order='F')
            b = np.multiply(mpml_z, a, order='F')
            c = np.multiply(mpml_x, b, order='F')
            c = c + kgrid.dt * np.multiply(lame_lambda, duydy, order='F') + kgrid.dt * np.multiply(chi, dduydydt, order='F')
            d = np.multiply(pml_y, c, order='F')
            e = np.multiply(mpml_z, d, order='F')
            sxx_split_y = np.multiply(mpml_x, e, order='F')

            # sxx_split_z = bsxfun(@times, mpml_y, bsxfun(@times, mpml_x, bsxfun(@times, pml_z, \
            #             bsxfun(@times, mpml_y, bsxfun(@times, mpml_x, bsxfun(@times, pml_z, sxx_split_z))) \
            #             + kgrid.dt * lame_lambda * duzdz \
            #             + kgrid.dt * chi * dduzdzdt)))
            sxx_split_z = mpml_y * mpml_x * pml_z * (mpml_y * mpml_x * pml_z * sxx_split_z +
                        kgrid.dt * lame_lambda * duzdz + kgrid.dt * chi * dduzdzdt)

            # syy_split_x = bsxfun(@times, mpml_z, bsxfun(@times, mpml_y, bsxfun(@times, pml_x, \
            #             bsxfun(@times, mpml_z, bsxfun(@times, mpml_y, bsxfun(@times, pml_x, syy_split_x))) \
            #             + kgrid.dt * (lame_lambda * duxdx + kgrid.dt * chi * dduxdxdt) )))
            syy_split_x = mpml_z * mpml_y * pml_x * (
                mpml_z * mpml_y * pml_x * syy_split_x +
                kgrid.dt * (lame_lambda * duxdx + chi * dduxdxdt))

            # syy_split_y = bsxfun(@times, mpml_x, bsxfun(@times, mpml_z, bsxfun(@times, pml_y, \
            #               bsxfun(@times, mpml_x, bsxfun(@times, mpml_z, bsxfun(@times, pml_y, syy_split_y))) \
            #             + kgrid.dt * (2 * mu + lame_lambda) * duydy \
            #             + kgrid.dt * (2 * eta + chi) * dduydydt )))
            syy_split_y = mpml_x * mpml_z * pml_y * (mpml_x * mpml_z * pml_y * syy_split_y + \
                                               kgrid.dt * (2.0 * mu + lame_lambda) * duydy + \
                                               kgrid.dt * (2.0 * eta + chi) * dduydydt)

            # syy_split_z = bsxfun(@times, mpml_y, bsxfun(@times, mpml_x, bsxfun(@times, pml_z, \
            #               bsxfun(@times, mpml_y, bsxfun(@times, mpml_x, bsxfun(@times, pml_z, syy_split_z))) \
            #             + kgrid.dt * lame_lambda * duzdz \
            #             + kgrid.dt * chi * dduzdzdt) ))
            syy_split_z = mpml_y * mpml_x * pml_z * (mpml_y * mpml_x * pml_z * syy_split_z +\
                                               kgrid.dt * lame_lambda * duzdz + \
                                               kgrid.dt * chi * dduzdzdt)

            # szz_split_x = bsxfun(@times, mpml_z, bsxfun(@times, mpml_y, bsxfun(@times, pml_x, \
            #             bsxfun(@times, mpml_z, bsxfun(@times, mpml_y, bsxfun(@times, pml_x, szz_split_x))) \
            #             + kgrid.dt * lame_lambda * duxdx\
            #             + kgrid.dt * chi * dduxdxdt)))
            szz_split_x = mpml_z * mpml_y * pml_x * (mpml_z * mpml_y * pml_x * szz_split_x + \
                                               kgrid.dt * lame_lambda * duxdx + \
                                               kgrid.dt * chi * dduxdxdt)

            # szz_split_y = bsxfun(@times, mpml_x, bsxfun(@times, mpml_z, bsxfun(@times, pml_y, \
            #             bsxfun(@times, mpml_x, bsxfun(@times, mpml_z, bsxfun(@times, pml_y, szz_split_y))) \
            #             + kgrid.dt * lame_lambda * duydy \
            #             + kgrid.dt * chi * dduydydt)))
            szz_split_y = mpml_x * mpml_z * pml_y * (mpml_x * mpml_z * pml_y * szz_split_y + \
                                               kgrid.dt * lame_lambda * duydy + \
                                               kgrid.dt * chi * dduydydt)

            # szz_split_z = bsxfun(@times, mpml_y, bsxfun(@times, mpml_x, bsxfun(@times, pml_z, \
            #             bsxfun(@times, mpml_y, bsxfun(@times, mpml_x, bsxfun(@times, pml_z, szz_split_z))) \
            #             + kgrid.dt * (2 * mu + lame_lambda) * duzdz \
            #             + kgrid.dt * (2 * eta + chi) * dduzdzdt)))
            szz_split_z = mpml_y * mpml_x * pml_z * (mpml_y * mpml_x * pml_z * szz_split_z + \
                                               kgrid.dt * (2.0 * mu + lame_lambda) * duzdz + \
                                               kgrid.dt * (2.0 * eta + chi) * dduzdzdt)

            # sxy_split_x = bsxfun(@times, mpml_z, bsxfun(@times, mpml_y_sgy, bsxfun(@times, pml_x_sgx, \
            #             bsxfun(@times, mpml_z, bsxfun(@times, mpml_y_sgy, bsxfun(@times, pml_x_sgx, sxy_split_x))) \
            #             + kgrid.dt * mu_sgxy * duydx \
            #             + kgrid.dt * eta_sgxy * dduydxdt)))
            sxy_split_x = mpml_z * mpml_y_sgy * pml_x_sgx * (mpml_z * mpml_y_sgy * pml_x_sgx * sxy_split_x + \
                                                             kgrid.dt * mu_sgxy * duydx + \
                                                             kgrid.dt * eta_sgxy * dduydxdt)

            # sxy_split_y = bsxfun(@times, mpml_z, bsxfun(@times, mpml_x_sgx, bsxfun(@times, pml_y_sgy, \
            #             bsxfun(@times, mpml_z, bsxfun(@times, mpml_x_sgx, bsxfun(@times, pml_y_sgy, sxy_split_y))) \
            #             + kgrid.dt * mu_sgxy * duxdy \
            #             + kgrid.dt * eta_sgxy * dduxdydt)))
            sxy_split_y = mpml_z * mpml_x_sgx * pml_y_sgy * (mpml_z * mpml_x_sgx * pml_y_sgy * sxy_split_y + \
                                               kgrid.dt * mu_sgxy * duxdy + \
                                               kgrid.dt * eta_sgxy * dduxdydt)

            # sxz_split_x = bsxfun(@times, mpml_y, bsxfun(@times, mpml_z_sgz, bsxfun(@times, pml_x_sgx, \
            #             bsxfun(@times, mpml_y, bsxfun(@times, mpml_z_sgz, bsxfun(@times, pml_x_sgx, sxz_split_x))) \
            #             + kgrid.dt * mu_sgxz * duzdx \
            #             + kgrid.dt * eta_sgxz * dduzdxdt)))
            sxz_split_x = mpml_y * mpml_z_sgz * pml_x_sgx * (mpml_y * mpml_z_sgz * pml_x_sgx * sxz_split_x + \
                                                               kgrid.dt * mu_sgxz * duzdx + \
                                                               kgrid.dt * eta_sgxz * dduzdxdt)

            # sxz_split_z = bsxfun(@times, mpml_y, bsxfun(@times, mpml_x_sgx, bsxfun(@times, pml_z_sgz, \
            #             bsxfun(@times, mpml_y, bsxfun(@times, mpml_x_sgx, bsxfun(@times, pml_z_sgz, sxz_split_z))) \
            #             + kgrid.dt * mu_sgxz * duxdz \
            #             + kgrid.dt * eta_sgxz * dduxdzdt)))
            sxz_split_z = mpml_y * mpml_x_sgx * pml_z_sgz * (mpml_y * mpml_x_sgx * pml_z_sgz * sxz_split_z + \
                                                               kgrid.dt * mu_sgxz * duxdz + \
                                                               kgrid.dt * eta_sgxz * dduxdzdt)

            # syz_split_y = bsxfun(@times, mpml_x, bsxfun(@times, mpml_z_sgz, bsxfun(@times, pml_y_sgy, \
            #             bsxfun(@times, mpml_x, bsxfun(@times, mpml_z_sgz, bsxfun(@times, pml_y_sgy, syz_split_y))) \
            #             + kgrid.dt * mu_sgyz * duzdy \
            #             + kgrid.dt * eta_sgyz * dduzdydt)))
            syz_split_y = mpml_x * mpml_z_sgz * pml_y_sgy * (mpml_x * mpml_z_sgz * pml_y_sgy * syz_split_y + \
                                                               kgrid.dt * mu_sgyz * duzdy + \
                                                               kgrid.dt * eta_sgyz * dduzdydt)

            # syz_split_z = bsxfun(@times, mpml_x, bsxfun(@times, mpml_y_sgy, bsxfun(@times, pml_z_sgz, \
            #             bsxfun(@times, mpml_x, bsxfun(@times, mpml_y_sgy, bsxfun(@times, pml_z_sgz, syz_split_z))) \
            #             + kgrid.dt * mu_sgyz * duydz \
            #             + kgrid.dt * eta_sgyz * dduydzdt)))
            syz_split_z = mpml_x * mpml_y_sgy * pml_z_sgz * (mpml_x * mpml_y_sgy * pml_z_sgz * syz_split_z + \
                                                               kgrid.dt * mu_sgyz * duydz + \
                                                               kgrid.dt * eta_sgyz * dduydzdt)

        else:

            # temp = 2.0 * mu + lame_lambda

            # update the normal and shear components of the stress tensor using
            # a lossless elastic model with a split-field multi-axial pml
            # sxx_split_x = bsxfun(@times, mpml_z,
            #                      bsxfun(@times, mpml_y,
            #                             bsxfun(@times, pml_x,
            #                                    bsxfun(@times, mpml_z,
            #                                           bsxfun(@times, mpml_y,
            #                                                  bsxfun(@times, pml_x, sxx_split_x))) + kgrid.dt * (2 * mu + lame_lambda) * duxdx ) ))
            sxx_split_x = mpml_z * (mpml_y * (pml_x * (mpml_z * (mpml_y * (pml_x * sxx_split_x)) + \
                                                     kgrid.dt * (2.0 * mu + lame_lambda) * duxdx)))

            # sxx_split_y = bsxfun(@times, mpml_x, bsxfun(@times, mpml_z, bsxfun(@times, pml_y, \
            #             bsxfun(@times, mpml_x, bsxfun(@times, mpml_z, bsxfun(@times, pml_y, sxx_split_y))) \
            #             + kgrid.dt * lame_lambda * duydy )))
            sxx_split_y = mpml_x * (mpml_z * (pml_y * (mpml_x * (mpml_z * (pml_y * sxx_split_y)) + \
                                               kgrid.dt * lame_lambda * duydy)))

            # sxx_split_z = bsxfun(@times, mpml_y, bsxfun(@times, mpml_x, bsxfun(@times, pml_z, \
            #             bsxfun(@times, mpml_y, bsxfun(@times, mpml_x, bsxfun(@times, pml_z, sxx_split_z))) \
            #             + kgrid.dt * lame_lambda * duzdz )))
            sxx_split_z = mpml_y * (mpml_x * (pml_z * (mpml_y * (mpml_x * (pml_z * sxx_split_z)) + \
                                               kgrid.dt * lame_lambda * duzdz)))

            # syy_split_x = bsxfun(@times, mpml_z, bsxfun(@times, mpml_y, bsxfun(@times, pml_x, \
            #             bsxfun(@times, mpml_z, bsxfun(@times, mpml_y, bsxfun(@times, pml_x, syy_split_x))) \
            #             + kgrid.dt * lame_lambda * duxdx)))
            syy_split_x = mpml_z * (mpml_y * (pml_x * (mpml_z * (mpml_y * (pml_x * syy_split_x)) + \
                                               kgrid.dt * lame_lambda * duxdx)))

            # syy_split_y = bsxfun(@times, mpml_x, bsxfun(@times, mpml_z, bsxfun(@times, pml_y, \
            #             bsxfun(@times, mpml_x, bsxfun(@times, mpml_z, bsxfun(@times, pml_y, syy_split_y))) \
            #             + kgrid.dt * (2 * mu + lame_lambda) * duydy )))
            syy_split_y = mpml_x * (mpml_z * (pml_y * (mpml_x * (mpml_z * (pml_y * syy_split_y)) + \
                                               kgrid.dt * (2.0 * mu + lame_lambda) * duydy)))

            # syy_split_z = bsxfun(@times, mpml_y, bsxfun(@times, mpml_x, bsxfun(@times, pml_z, \
            #             bsxfun(@times, mpml_y, bsxfun(@times, mpml_x, bsxfun(@times, pml_z, syy_split_z))) \
            #             + kgrid.dt * lame_lambda * duzdz )))
            syy_split_z = mpml_y * (mpml_x * (pml_z * (mpml_y * (mpml_x * (pml_z * syy_split_z)) + \
                                               kgrid.dt * lame_lambda * duzdz)))

            # szz_split_x = bsxfun(@times, mpml_z, bsxfun(@times, mpml_y, bsxfun(@times, pml_x, \
            #             bsxfun(@times, mpml_z, bsxfun(@times, mpml_y, bsxfun(@times, pml_x, szz_split_x))) \
            #             + kgrid.dt * lame_lambda * duxdx)))
            szz_split_x = mpml_z * (mpml_y * (pml_x * (mpml_z * (mpml_y * (pml_x * szz_split_x)) + \
                                               kgrid.dt * lame_lambda * duxdx)))

            # szz_split_y = bsxfun(@times, mpml_x, bsxfun(@times, mpml_z, bsxfun(@times, pml_y, \
            #             bsxfun(@times, mpml_x, bsxfun(@times, mpml_z, bsxfun(@times, pml_y, szz_split_y))) \
            #             + kgrid.dt * lame_lambda * duydy )))
            szz_split_y = mpml_x * (mpml_z * (pml_y * (mpml_x * (mpml_z * (pml_y * szz_split_y)) + \
                                               kgrid.dt * lame_lambda * duydy)))

            # szz_split_z = bsxfun(@times, mpml_y, bsxfun(@times, mpml_x, bsxfun(@times, pml_z, \
            #             bsxfun(@times, mpml_y, bsxfun(@times, mpml_x, bsxfun(@times, pml_z, szz_split_z))) \
            #             + kgrid.dt * (2 * mu + lame_lambda) * duzdz )))
            szz_split_z = mpml_y * (mpml_x * (pml_z * (mpml_y * (mpml_x * (pml_z * szz_split_z)) + \
                                               kgrid.dt * (2.0 * mu + lame_lambda) * duzdz)))

            # sxy_split_x = bsxfun(@times, mpml_z,
            #                      bsxfun(@times, mpml_y_sgy,
            #                             bsxfun(@times, pml_x_sgx, \
            #                                    bsxfun(@times, mpml_z,
            #                                           bsxfun(@times, mpml_y_sgy,
            #                                                  bsxfun(@times, pml_x_sgx, sxy_split_x))) +\
            #                                                  kgrid.dt * mu_sgxy * duydx)))
            sxy_split_x = mpml_z * (mpml_y_sgy * (pml_x_sgx * (mpml_z * (mpml_y_sgy * (pml_x_sgx * sxy_split_x)) + \
                                                               kgrid.dt * mu_sgxy * duydx)))

            # sxy_split_y = bsxfun(@times, mpml_z, bsxfun(@times, mpml_x_sgx, bsxfun(@times, pml_y_sgy, \
            #             bsxfun(@times, mpml_z, bsxfun(@times, mpml_x_sgx, bsxfun(@times, pml_y_sgy, sxy_split_y))) \
            #             + kgrid.dt * mu_sgxy * duxdy)))
            sxy_split_y = mpml_z * (mpml_x_sgx * (pml_y_sgy * (mpml_z * (mpml_x_sgx * (pml_y_sgy * sxy_split_y)) + \
                                                               kgrid.dt * mu_sgxy * duxdy)))

            # sxz_split_x = bsxfun(@times, mpml_y, bsxfun(@times, mpml_z_sgz, bsxfun(@times, pml_x_sgx, \
            #             bsxfun(@times, mpml_y, bsxfun(@times, mpml_z_sgz, bsxfun(@times, pml_x_sgx, sxz_split_x))) \
            #             + kgrid.dt * mu_sgxz * duzdx)))
            sxz_split_x = mpml_y * (mpml_z_sgz * (pml_x_sgx * (mpml_y * (mpml_z_sgz * (pml_x_sgx * sxz_split_x)) + \
                                                               kgrid.dt * mu_sgxz * duzdx)))

            # sxz_split_z = bsxfun(@times, mpml_y, bsxfun(@times, mpml_x_sgx, bsxfun(@times, pml_z_sgz, \
            #             bsxfun(@times, mpml_y, bsxfun(@times, mpml_x_sgx, bsxfun(@times, pml_z_sgz, sxz_split_z))) \
            #             + kgrid.dt * mu_sgxz * duxdz)))
            sxz_split_z = mpml_y * (mpml_x_sgx * (pml_z_sgz * (mpml_y * (mpml_x_sgx * (pml_z_sgz * sxz_split_z)) + \
                                                               kgrid.dt * mu_sgxz * duxdz)))

            # syz_split_y = bsxfun(@times, mpml_x, bsxfun(@times, mpml_z_sgz, bsxfun(@times, pml_y_sgy, \
            #             bsxfun(@times, mpml_x, bsxfun(@times, mpml_z_sgz, bsxfun(@times, pml_y_sgy, syz_split_y))) \
            #             + kgrid.dt * mu_sgyz * duzdy)))
            syz_split_y = mpml_x * (mpml_z_sgz * (pml_y_sgy * (mpml_x * (mpml_z_sgz * (pml_y_sgy * syz_split_y)) + \
                                                               kgrid.dt * mu_sgyz * duzdy)))

            # syz_split_z = bsxfun(@times, mpml_x,
            #                      bsxfun(@times, mpml_y_sgy,
            #                             bsxfun(@times, pml_z_sgz,
            #                                    bsxfun(@times, mpml_x,
            #                                           bsxfun(@times, mpml_y_sgy,
            #                                                  bsxfun(@times, pml_z_sgz, syz_split_z))) + kgrid.dt * mu_sgyz * duydz)))
            syz_split_z  = mpml_x * (mpml_y_sgy * (pml_z_sgz * (mpml_x * (mpml_y_sgy * (pml_z_sgz * syz_split_z)) + \
                                                                kgrid.dt * mu_sgyz * duydz)))


        # add in the pre-scaled stress source terms

        if hasattr(k_sim, 's_source_sig_index'):
            if isinstance(k_sim.s_source_sig_index, str):
                if k_sim.s_source_sig_index == ':':
                    print("converting s_source_sig_index")
                    if (k_sim.source_sxx is not False):
                        k_sim.s_source_sig_index = np.shape(source.sxx)[0]
                    elif (k_sim.source_syy is not False):
                        k_sim.s_source_sig_index = np.shape(source.syy)[0]
                    elif (k_sim.source_szz is not False):
                        k_sim.s_source_sig_index = np.shape(source.szz)[0]
                    elif (k_sim.source_sxy is not False):
                        k_sim.s_source_sig_index = np.shape(source.sxy)[0]
                    elif (k_sim.source_sxy is not False):
                        k_sim.s_source_sig_index = np.shape(source.sxz)[0]
                    elif (k_sim.source_syz is not False):
                        k_sim.s_source_sig_index = np.shape(source.syz)[0]
                    else:
                        raise RuntimeError('Need to set s_source_sig_index')

        if (k_sim.source_sxx is not False and t_index < np.size(source.sxx)):

            # if (t_index == 0):
                # print(np.shape(k_sim.source.sxx), np.shape(k_sim.s_source_pos_index), np.shape(k_sim.s_source_sig_index) )

            if (source.s_mode == 'dirichlet'):
                # enforce the source values as a dirichlet boundary condition
                sxx_split_x[np.unravel_index(k_sim.s_source_pos_index, sxx_split_x.shape, order='F')] = k_sim.source.sxx[k_sim.s_source_sig_index, t_index]
                sxx_split_y[np.unravel_index(k_sim.s_source_pos_index, sxx_split_y.shape, order='F')] = k_sim.source.sxx[k_sim.s_source_sig_index, t_index]
                sxx_split_z[np.unravel_index(k_sim.s_source_pos_index, sxx_split_z.shape, order='F')] = k_sim.source.sxx[k_sim.s_source_sig_index, t_index]
            else:
                # add the source values to the existing field values
                sxx_split_x[np.unravel_index(k_sim.s_source_pos_index, sxx_split_x.shape, order='F')] = sxx_split_x[np.unravel_index(k_sim.s_source_pos_index, sxx_split_x.shape, order='F')] + \
                  k_sim.source.sxx[k_sim.s_source_sig_index, t_index]
                sxx_split_y[np.unravel_index(k_sim.s_source_pos_index, sxx_split_y.shape, order='F')] = sxx_split_y[np.unravel_index(k_sim.s_source_pos_index, sxx_split_y.shape, order='F')] + \
                  k_sim.source.sxx[k_sim.s_source_sig_index, t_index]
                sxx_split_z[np.unravel_index(k_sim.s_source_pos_index, sxx_split_z.shape, order='F')] = sxx_split_z[np.unravel_index(k_sim.s_source_pos_index, sxx_split_z.shape, order='F')] + \
                  k_sim.source.sxx[k_sim.s_source_sig_index, t_index]

        if (k_sim.source_syy is not False and t_index < np.size(source.syy)):
            if (source.s_mode == 'dirichlet'):
                # enforce the source values as a dirichlet boundary condition
                syy_split_x[np.unravel_index(k_sim.s_source_pos_index, syy_split_x.shape, order='F')] = k_sim.source.syy[k_sim.s_source_sig_index, t_index]
                syy_split_y[np.unravel_index(k_sim.s_source_pos_index, syy_split_y.shape, order='F')] = k_sim.source.syy[k_sim.s_source_sig_index, t_index]
                syy_split_z[np.unravel_index(k_sim.s_source_pos_index, syy_split_z.shape, order='F')] = k_sim.source.syy[k_sim.s_source_sig_index, t_index]
            else:
                # add the source values to the existing field values
                syy_split_x[np.unravel_index(k_sim.s_source_pos_index, syy_split_x.shape, order='F')] = syy_split_x[np.unravel_index(k_sim.s_source_pos_index, syy_split_x.shape, order='F')] + \
                  k_sim.source.syy[k_sim.s_source_sig_index, t_index]
                syy_split_y[np.unravel_index(k_sim.s_source_pos_index, syy_split_y.shape, order='F')] = syy_split_y[np.unravel_index(k_sim.s_source_pos_index, syy_split_y.shape, order='F')] + \
                  k_sim.source.syy[k_sim.s_source_sig_index, t_index]
                syy_split_z[np.unravel_index(k_sim.s_source_pos_index, syy_split_z.shape, order='F')] = syy_split_z[np.unravel_index(k_sim.s_source_pos_index, syy_split_z.shape, order='F')] + \
                  k_sim.source.syy[k_sim.s_source_sig_index, t_index]

        if (k_sim.source_szz is not False and t_index < np.size(source.szz)):
            if (source.s_mode == 'dirichlet'):
                # enforce the source values as a dirichlet boundary condition
                szz_split_x[np.unravel_index(k_sim.s_source_pos_index, szz_split_x.shape, order='F')] = k_sim.source.szz[k_sim.s_source_sig_index, t_index]
                szz_split_y[np.unravel_index(k_sim.s_source_pos_index, szz_split_y.shape, order='F')] = k_sim.source.szz[k_sim.s_source_sig_index, t_index]
                szz_split_z[np.unravel_index(k_sim.s_source_pos_index, szz_split_z.shape, order='F')] = k_sim.source.szz[k_sim.s_source_sig_index, t_index]
            else:
                # # add the source values to the existing field values
                szz_split_x[np.unravel_index(k_sim.s_source_pos_index, szz_split_x.shape, order='F')] = szz_split_x[np.unravel_index(k_sim.s_source_pos_index, szz_split_x.shape, order='F')] + \
                  k_sim.source.szz[k_sim.s_source_sig_index, t_index]
                szz_split_y[np.unravel_index(k_sim.s_source_pos_index, szz_split_y.shape, order='F')] = szz_split_y[np.unravel_index(k_sim.s_source_pos_index, szz_split_y.shape, order='F')] + \
                  k_sim.source.szz[k_sim.s_source_sig_index, t_index]
                szz_split_z[np.unravel_index(k_sim.s_source_pos_index, szz_split_z.shape, order='F')] = szz_split_z[np.unravel_index(k_sim.s_source_pos_index, szz_split_z.shape, order='F')] + \
                  k_sim.source.szz[k_sim.s_source_sig_index, t_index]

        if (k_sim.source_sxy is not False and t_index < np.size(source.sxy)):
            if (source.s_mode == 'dirichlet'):
                # enforce the source values as a dirichlet boundary condition
                sxy_split_x[np.unravel_index(k_sim.s_source_pos_index, sxy_split_x.shape, order='F')] = k_sim.source.sxy[k_sim.s_source_sig_index, t_index]
                sxy_split_y[np.unravel_index(k_sim.s_source_pos_index, sxy_split_y.shape, order='F')] = k_sim.source.sxy[k_sim.s_source_sig_index, t_index]
                # add the source values to the existing field values
            else:
                sxy_split_x[np.unravel_index(k_sim.s_source_pos_index, sxy_split_x.shape, order='F')] = sxy_split_x[np.unravel_index(k_sim.s_source_pos_index, sxy_split_x.shape, order='F')] + \
                  k_sim.source.sxy[k_sim.s_source_sig_index, t_index]
                sxy_split_y[np.unravel_index(k_sim.s_source_pos_index, sxy_split_y.shape, order='F')] = sxy_split_y[np.unravel_index(k_sim.s_source_pos_index, sxy_split_y.shape, order='F')] + \
                  k_sim.source.sxy[k_sim.s_source_sig_index, t_index]

        if (k_sim.source_sxz is not False and t_index < np.size(source.sxz)):
            if (source.s_mode == 'dirichlet'):
                # enforce the source values as a dirichlet boundary condition
                sxz_split_x[np.unravel_index(k_sim.s_source_pos_index, sxz_split_x.shape, order='F')] = k_sim.source.sxz[k_sim.s_source_sig_index, t_index]
                sxz_split_z[np.unravel_index(k_sim.s_source_pos_index, sxz_split_z.shape, order='F')] = k_sim.source.sxz[k_sim.s_source_sig_index, t_index]
            else:
                # add the source values to the existing field values
                sxz_split_x[np.unravel_index(k_sim.s_source_pos_index, sxz_split_x.shape, order='F')] = sxz_split_x[np.unravel_index(k_sim.s_source_pos_index, sxz_split_x.shape, order='F')] + \
                  k_sim.source.sxz[k_sim.s_source_sig_index, t_index]
                sxz_split_z[np.unravel_index(k_sim.s_source_pos_index, sxz_split_z.shape, order='F')] = sxz_split_z[np.unravel_index(k_sim.s_source_pos_index, sxz_split_z.shape, order='F')] + \
                  k_sim.source.sxz[k_sim.s_source_sig_index,  t_index]

        if (k_sim.source_syz is not False and t_index < np.size(source.syz)):
            if (source.s_mode == 'dirichlet'):
                # enforce the source values as a dirichlet boundary condition
                syz_split_y[np.unravel_index(k_sim.s_source_pos_index, syz_split_y.shape, order='F')] = k_sim.source.syz[k_sim.s_source_sig_index, t_index]
                syz_split_z[np.unravel_index(k_sim.s_source_pos_index, syz_split_z.shape, order='F')] = k_sim.source.syz[k_sim.s_source_sig_index, t_index]
            else:
                # add the source values to the existing field values
                syz_split_y[np.unravel_index(k_sim.s_source_pos_index, syz_split_y.shape, order='F')] = syz_split_y[np.unravel_index(k_sim.s_source_pos_index, syz_split_y.shape, order='F')] + \
                  k_sim.source.syz[k_sim.s_source_sig_index, t_index]
                syz_split_z[np.unravel_index(k_sim.s_source_pos_index, syz_split_z.shape, order='F')] = syz_split_z[np.unravel_index(k_sim.s_source_pos_index, syz_split_z.shape, order='F')] + \
                  k_sim.source.syz[k_sim.s_source_sig_index, t_index]

        if checking:
            if (t_index == load_index):

                print("k_sim.s_source_pos_index:", k_sim.s_source_pos_index)

                print("k_sim.source.sxx:", k_sim.source.sxx)

                print("np.max(k_sim.s_source_sig_index):", np.max(k_sim.s_source_sig_index))
                print("np.sum(k_sim.s_source_sig_index)", np.sum(k_sim.s_source_sig_index))
                print("np.shape(k_sim.s_source_sig_index)", np.shape(k_sim.s_source_sig_index))
                print("sxx_split_x[np.unravel_index(k_sim.s_source_pos_index, sxx_split_x.shape, order='F')]",
                      sxx_split_x[np.unravel_index(k_sim.s_source_pos_index, sxx_split_x.shape, order='F')])

                diff0 = np.max(np.abs(sxx_split_x - mat_sxx_split_x))
                if (diff0 > tol):
                    print("\tsxx_split_x is incorrect:" + diff0)
                diff0 = np.max(np.abs(sxx_split_y - mat_sxx_split_y))
                if (diff0 > tol):
                    print("\tsxx_split_y is incorrect:" + diff0)
                diff0 = np.max(np.abs(sxy_split_y - mat_sxy_split_y))
                if (diff0 > tol):
                    print("\tsxy_split_y is incorrect:" + diff0)
                diff0 = np.max(np.abs(sxy_split_x - mat_sxy_split_x))
                if (diff0 > tol):
                    print("\tsxy_split_x is incorrect:" + diff0)


        # compute pressure from the normal components of the stress
        p = -(sxx_split_x + sxx_split_y + sxx_split_z +
              syy_split_x + syy_split_y + syy_split_z +
              szz_split_x + szz_split_y + szz_split_z) / 3.0

        # update index for data storage
        file_index: int = t_index - sensor.record_start_index

        # extract required sensor data from the pressure and particle velocity
        # fields if the number of time steps elapsed is greater than
        # sensor.record_start_index (defaults to 1) - why not zero?
        if ((options.use_sensor is not False) and (not options.elastic_time_rev) and (t_index >= k_sim.sensor.record_start_index)):

            # print("FAIL")

            # update index for data storage
            file_index: int = t_index - sensor.record_start_index

            # store the acoustic pressure if using a transducer object
            if k_sim.transducer_sensor:
                raise TypeError('Using a kWaveTransducer for output is not currently supported.')

            extract_options = dotdict({'record_u_non_staggered': k_sim.record.u_non_staggered,
                                       'record_u_split_field': k_sim.record.u_split_field,
                                       'record_I': k_sim.record.I,
                                       'record_I_avg': k_sim.record.I_avg,
                                       'binary_sensor_mask': k_sim.binary_sensor_mask,
                                       'record_p': k_sim.record.p,
                                       'record_p_max': k_sim.record.p_max,
                                       'record_p_min': k_sim.record.p_min,
                                       'record_p_rms': k_sim.record.p_rms,
                                       'record_p_max_all': k_sim.record.p_max_all,
                                       'record_p_min_all': k_sim.record.p_min_all,
                                       'record_u': k_sim.record.u,
                                       'record_u_max': k_sim.record.u_max,
                                       'record_u_min': k_sim.record.u_min,
                                       'record_u_rms': k_sim.record.u_rms,
                                       'record_u_max_all': k_sim.record.u_max_all,
                                       'record_u_min_all': k_sim.record.u_min_all,
                                       'compute_directivity': False,
                                       'use_cuboid_corners': options.cuboid_corners})

            # run sub-function to extract the required data
            sensor_data = extract_sensor_data(3, sensor_data, file_index, k_sim.sensor_mask_index,
                                              extract_options, k_sim.record, p, ux_sgx, uy_sgy, uz_sgz)

            # check stream to disk option
            if options.stream_to_disk:
                raise TypeError('"StreamToDisk" input is not currently supported.')



        # # plot data if required
        # if options.plot_sim and (rem(t_index, plot_freq) == 0 or t_index == 1 or t_index == index_end):

        #     # update progress bar
        #     waitbar(t_index/kgrid.Nt, pbar)
        #     drawnow

        #     # ensure p is cast as a CPU variable and remove the PML from the
        #     # plot if required
        #     if (data_cast == 'gpuArray'):
        #         sii_plot = double(gather(p(x1:x2, y1:y2, z1:z2)))
        #         sij_plot = double(gather((\
        #         sxy_split_x(x1:x2, y1:y2, z1:z2) + sxy_split_y(x1:x2, y1:y2, z1:z2) + \
        #         sxz_split_x(x1:x2, y1:y2, z1:z2) + sxz_split_z(x1:x2, y1:y2, z1:z2) + \
        #         syz_split_y(x1:x2, y1:y2, z1:z2) + syz_split_z(x1:x2, y1:y2, z1:z2) )/3))
        #     else:
        #         sii_plot = double(p(x1:x2, y1:y2, z1:z2))
        #         sij_plot = double((\
        #         sxy_split_x(x1:x2, y1:y2, z1:z2) + sxy_split_y(x1:x2, y1:y2, z1:z2) + \
        #         sxz_split_x(x1:x2, y1:y2, z1:z2) + sxz_split_z(x1:x2, y1:y2, z1:z2) + \
        #         syz_split_y(x1:x2, y1:y2, z1:z2) + syz_split_z(x1:x2, y1:y2, z1:z2) )/3)


        #     # update plot scale if set to automatic or log
        #     if options.plot_scale_auto or options.plot_scale_log:
        #         kspaceFirstOrder_adjustPlotScale

        #     # add display mask onto plot
        #     if (display_mask == 'default'):
        #         sii_plot(sensor.mask(x1:x2, y1:y2, z1:z2) != 0) = plot_scale(2)
        #         sij_plot(sensor.mask(x1:x2, y1:y2, z1:z2) != 0) = plot_scale(end)
        #     elif not (display_mask == 'off'):
        #         sii_plot(display_mask(x1:x2, y1:y2, z1:z2) != 0) = plot_scale(2)
        #         sij_plot(display_mask(x1:x2, y1:y2, z1:z2) != 0) = plot_scale(end)


        #     # update plot
        #     planeplot(scale * kgrid.x_vec(x1:x2),
        #               scale * kgrid.y_vec(y1:y2),
        #               scale * kgrid.z_vec(z1:z2),
        #               sii_plot,
        #               sij_plot, '',
        #               plot_scale,
        #               prefix,
        #               COLOR_MAP)

        #     # save movie frames if required
        #     if options.record_movie:

        #         # set background color to white
        #         set(gcf, 'Color', [1 1 1])

        #         # save the movie frame
        #         writeVideo(video_obj, getframe(gcf))



            # # update variable used for timing variable to exclude the first
            # # time step if plotting is enabled
            # if t_index == 0:
            #     loop_start_time = clock


    # update command line status
    print('\tsimulation completed in', scale_time(TicToc.toc()))

    # =========================================================================
    # CLEAN UP
    # =========================================================================

    # print("bounds:", record.x1_inside, record.x2_inside, record.y1_inside, record.y2_inside, record.z1_inside, record.z2_inside)

    if not options.cuboid_corners:
        # save the final acoustic pressure if required
        if k_sim.record.p_final or options.elastic_time_rev:
            sensor_data.p_final = p[record.x1_inside:record.x2_inside,
                                    record.y1_inside:record.y2_inside,
                                    record.z1_inside:record.z2_inside]
        # save the final particle velocity if required
        if k_sim.record.u_final:
            sensor_data.ux_final = ux_sgx[record.x1_inside:record.x2_inside,
                                          record.y1_inside:record.y2_inside,
                                          record.z1_inside:record.z2_inside]
            sensor_data.uy_final = uy_sgy[record.x1_inside:record.x2_inside,
                                          record.y1_inside:record.y2_inside,
                                          record.z1_inside:record.z2_inside]
            sensor_data.uz_final = uz_sgz[record.x1_inside:record.x2_inside,
                                          record.y1_inside:record.y2_inside,
                                          record.z1_inside:record.z2_inside]
    else:
        # save the final acoustic pressure if required
        if k_sim.record.p_final or options.elastic_time_rev:
            sensor_data.append(dotdict({'p_final': p[record.x1_inside:record.x2_inside,
                                    record.y1_inside:record.y2_inside,
                                    record.z1_inside:record.z2_inside]}))
            # save the final particle velocity if required
            if k_sim.record.u_final:
                i: int = len(sensor_data) - 1
                sensor_data[i].ux_final = ux_sgx[record.x1_inside:record.x2_inside,
                                              record.y1_inside:record.y2_inside,
                                              record.z1_inside:record.z2_inside]
                sensor_data[i].uy_final = uy_sgy[record.x1_inside:record.x2_inside,
                                              record.y1_inside:record.y2_inside,
                                              record.z1_inside:record.z2_inside]
                sensor_data[i].uz_final = uz_sgz[record.x1_inside:record.x2_inside,
                                              record.y1_inside:record.y2_inside,
                                              record.z1_inside:record.z2_inside]
        elif k_sim.record.u_final:
            sensor_data.append(dotdict({'ux_final': ux_sgx[record.x1_inside:record.x2_inside,
                                                           record.y1_inside:record.y2_inside,
                                                           record.z1_inside:record.z2_inside],
                                        'uy_final': uy_sgy[record.x1_inside:record.x2_inside,
                                                           record.y1_inside:record.y2_inside,
                                                           record.z1_inside:record.z2_inside],
                                        'uz_final': uz_sgz[record.x1_inside:record.x2_inside,
                                                           record.y1_inside:record.y2_inside,
                                                           record.z1_inside:record.z2_inside]}))


        # logging.log(logging.WARN, "  not done if there are cuboid corners")

    # # run subscript to cast variables back to double precision if required
    # if options.data_recast:
    #     kspaceFirstOrder_dataRecast

    # # run subscript to compute and save intensity values
    if options.use_sensor and not options.elastic_time_rev and (k_sim.record.I or k_sim.record.I_avg):
        save_intensity_options = dotdict({'record_I_avg': k_sim.record.I_avg,
                                          'record_p': k_sim.record.p,
                                          'record_I': k_sim.record.I,
                                          'record_u_non_staggered': k_sim.record.u_non_staggered,
                                          'use_cuboid_corners': options.cuboid_corners})
        sensor_data = save_intensity(kgrid, sensor_data, save_intensity_options)

    # reorder the sensor points if a binary sensor mask was used for Cartesian
    # sensor mask nearest neighbour interpolation (this is performed after
    # recasting as the GPU toolboxes do not all support this subscript)
    if options.use_sensor and k_sim.reorder_data:
        print("reorder?")
        sensor_data = reorder_sensor_data(kgrid, sensor, deepcopy(sensor_data))

    # filter the recorded time domain pressure signals if transducer filter
    # parameters are given
    if options.use_sensor and (not options.elastic_time_rev) and k_sim.sensor.frequency_response is not None:
        sensor_data.p = gaussian_filter(sensor_data.p, 1.0 / kgrid.dt,
                                        k_sim.sensor.frequency_response[0], k_sim.sensor.frequency_response[1])

    # # reorder the sensor points if cuboid corners is used (outputs are indexed
    # # as [X, Y, Z, T] or [X, Y, Z] rather than [sensor_index, time_index]
    # num_stream_time_points: int = k_sim.kgrid.Nt - k_sim.sensor.record_start_index
    # if options.cuboid_corners:
    #     print("cuboid corners?")
    #     time_info = dotdict({'num_stream_time_points': num_stream_time_points,
    #                          'num_recorded_time_points': k_sim.num_recorded_time_points,
    #                          'stream_to_disk': options.stream_to_disk})
    #     cuboid_info = dotdict({'record_p': k_sim.record.p,
    #                            'record_p_rms': k_sim.record.p_rms,
    #                            'record_p_max': k_sim.record.p_max,
    #                            'record_p_min': k_sim.record.p_min,
    #                            'record_p_final': k_sim.record.p_final,
    #                            'record_p_max_all': k_sim.record.p_max_all,
    #                            'record_p_min_all': k_sim.record.p_min_all,
    #                            'record_u': k_sim.record.u,
    #                            'record_u_non_staggered': k_sim.record.u_non_staggered,
    #                            'record_u_rms': k_sim.record.u_rms,
    #                            'record_u_max': k_sim.record.u_max,
    #                            'record_u_min': k_sim.record.u_min,
    #                            'record_u_final': k_sim.record.u_final,
    #                            'record_u_max_all': k_sim.record.u_max_all,
    #                            'record_u_min_all': k_sim.record.u_min_all,
    #                            'record_I': k_sim.record.I,
    #                            'record_I_avg': k_sim.record.I_avg})
    #     sensor_data = reorder_cuboid_corners(k_sim.kgrid, k_sim.record, sensor_data, time_info, cuboid_info, verbose=True)

    if options.elastic_time_rev:
        # if computing time reversal, reassign sensor_data.p_final to sensor_data
        # sensor_data = sensor_data.p_final
        raise NotImplementedError("elastic_time_rev is not implemented")
    elif not options.use_sensor:
        # if sensor is not used, return empty sensor data
        print("not options.use_sensor: returns None ->", options.use_sensor)
        sensor_data = None
    elif (sensor.record is None) and (not options.cuboid_corners):
        # if sensor.record is not given by the user, reassign sensor_data.p to sensor_data
        print("reassigns. Not sure if there is a check for whether this exists though")
        sensor_data = sensor_data.p
    else:
        pass

    # update command line status
    print('\ttotal computation time', scale_time(TicToc.toc()))

    # # switch off log
    # if options.create_log:
    #     diary off

    return sensor_data


# def planeplot(x_vec, y_vec, z_vec, s_normal_plot, s_shear_plot, data_title, plot_scale, prefix, color_map):
#     """
#     Subfunction to produce a plot of the elastic wavefield
#     """

#     # plot normal stress
#     subplot(2, 3, 1)
#     imagesc(y_vec, x_vec, squeeze(s_normal_plot(:, :, round(end/2))), plot_scale(1:2))
#     title('Normal Stress (x-y plane)'), axis image

#     subplot(2, 3, 2)
#     imagesc(z_vec, x_vec, squeeze(s_normal_plot(:, round(end/2), :)), plot_scale(1:2))
#     title('Normal Stress  (x-z plane)'), axis image

#     subplot(2, 3, 3)
#     imagesc(z_vec, y_vec, squeeze(s_normal_plot(round(end/2), :, :)), plot_scale(1:2))
#     title('Normal Stress  (y-z plane)'), axis image

#     # plot shear stress
#     subplot(2, 3, 4)
#     imagesc(y_vec, x_vec, squeeze(s_shear_plot(:, :, round(end/2))), plot_scale(end - 1:end))
#     title('Shear Stress (x-y plane)'), axis image

#     subplot(2, 3, 5)
#     imagesc(z_vec, x_vec, squeeze(s_shear_plot(:, round(end/2), :)), plot_scale(end - 1:end))
#     title('Shear Stress (x-z plane)'), axis image

#     subplot(2, 3, 6)
#     imagesc(z_vec, y_vec, squeeze(s_shear_plot(round(end/2), :, :)), plot_scale(end - 1:end))
#     title('Shear Stress (y-z plane)'), axis image

#     xlabel(['(All axes in ' prefix 'm)'])
#     colormap(color_map)
#     drawnow