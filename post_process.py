import numpy as np
# import numpy.linalg as la
import h5py

import pyvista as pv
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from kwave.utils.filters import extract_amp_phase

# from scipy.io import loadmat, savemat
# from scipy.interpolate import interp1d, interp2d, interpn, RegularGridInterpolator

# import nibabel as nib
import meshio
import pyvista
from skimage import measure

# from tqdm import tqdm

# import vtk


class PostProcess():

    def __init__(self, filename: str, freq: float, verbose: bool = False):
        """initialise class with the filename"""
        from pathlib import Path
        self.filename = filename
        self.freq = freq
        self.verbose = False
        self.ext = Path(filename).suffixes

    def hdf5_loader(self, filename):
        """loads data from a hdf5 file"""
        with h5py.File(self.filename, "r") as f:
            # Print all root level object names (aka keys)
            # these can be group or dataset names
            if self.verbose:
                print("Keys: %s" % f.keys())

        self.Nt = np.squeeze(np.array(h5py.File(filename, mode='r')["Nt"]))
        self.Nx = np.squeeze(np.array(h5py.File(filename, mode='r')["Nx"]))
        self.Ny = np.squeeze(np.array(h5py.File(filename, mode='r')["Ny"]))
        self.Nz = np.squeeze(np.array(h5py.File(filename, mode='r')["Nz"]))

        pml_x_size = np.squeeze(np.array(h5py.File(filename, mode='r')["pml_x_size"]))
        pml_y_size = np.squeeze(np.array(h5py.File(filename, mode='r')["pml_y_size"]))
        pml_z_size = np.squeeze(np.array(h5py.File(filename, mode='r')["pml_z_size"]))

        sensor_data = np.array(h5py.File(filename, mode='r')["p"])[0].T

        self.dt = np.squeeze(np.array(h5py.File(filename, mode='r')["dt"]))

        self.dx = np.squeeze(np.array(h5py.File(filename, mode='r')["dx"]))
        self.dy = np.squeeze(np.array(h5py.File(filename, mode='r')["dy"]))
        self.dz = np.squeeze(np.array(h5py.File(filename, mode='r')["dz"]))

        self.fs = 1.0 / self.dt

        self.Nx = int(self.Nx - 2 * pml_x_size)
        self.Ny = int(self.Ny - 2 * pml_y_size)
        self.Nz = int(self.Nz - 2 * pml_z_size)

        return sensor_data

    def get_temporal_average(self, dynamic_data, dim=1, fft_padding=1, window='Rectangular', order='C'):
        """get Fourier coefficients"""
        amp, _, _ = extract_amp_phase(dynamic_data, self.fs, self.freq, dim=dim,
                                      fft_padding=fft_padding, window=window)
        p = np.reshape(amp, (self.Nx, self.Ny, self.Nz), order=order)
        return p

    def get_spatial_average(self, plane, value, orientation='xy', plot=False):
        """In a given plane, perpendicular to the beam axis, calculate the
        spatial average of the acoustic field within a closed iso-pressure
        contour. The area defined by the contour is called the beam area and
        is denoted as $A_{b,6}$ for the -6dB beam area. The beam area is
        expressed in units of metre squared. Has a helper function which
        computes the area enclosed rather than sum voxels

        Notes
        -----
        The position of the plane is not specified.

        Parameters
        ----------
        plane
            The plane at which the spatial averaging is made
        value: `float`
            The second parameter.
        orientation : :obj:`str`, optional
            The second parameter.
        plot : :obj:`bool`, optional
            Plots the beam area

        Returns
        -------
        float
            The spatial average in the plane above a value.

        """
        # Todo: how to ensure that the plane is oriented in the correct way
        # TODO: Integrate values within the largest contour
        from scipy.integrate import simpson

        def area(vs):
            a = 0
            x0, y0 = vs[0]
            v = vs[1:-1]
            for count, val in enumerate(v):
                x1, y1 = val
                dx = x1 - x0
                dy = y1 - y0
                a += 0.5 * (y0 * dx - x0 * dy)
                x0 = x1
                y0 = y1
            return a

        spatial_average = 0.0
        a = 0
        plane0 = np.where(plane > value, plane, 0.0)
        n = np.count_nonzero(plane0)
        dA = 1.0
        if (plot and not np.isnan(spatial_average)):
            if ((orientation.lower() == 'xy') or (orientation.lower() == 'yx')):
                nx, ny = np.shape(plane)
                x = np.linspace(0, self.dx * (float(nx) - 1.0), self.Nx)
                y = np.linspace(0, self.dy * (float(ny) - 1.0), self.Ny)
            fig, ax = plt.subplots()
            im = ax.pcolor(x, y, plane0)
            cs = ax.contour(x, y, plane, levels=[value], colors='red')
            cbar = fig.colorbar(im, ax=ax)
            cbar.add_lines(cs)
            contour = cs.collections[0]
            vs = contour.get_paths()[0].vertices
            a = np.abs( area(vs) )
            if (len(contour.get_paths()) != 1):
                msg = "There are more than one contours"
                print(msg)
                ax.scatter(vs[:, 0], vs[:, 1])
            plt.show()

        if ((orientation.lower() == 'xy') or (orientation.lower() == 'yx')):
            nx, ny = np.shape(plane)
            x = np.linspace(0, self.dx * (float(nx) - 1.0), self.Nx)
            y = np.linspace(0, self.dy * (float(ny) - 1.0), self.Ny)
            dA = self.dx * self.dy
            spatial_average = simpson(simpson(plane0, dx=self.dy), dx=self.dx)
            spatial_average1 = simpson(simpson(plane0, dx=1), dx=1)

            m1 = np.zeros((self.Nx,))
            for i in np.arange(self.Nx):
                m1[i] = np.count_nonzero(plane0[:, i])
            v1 = np.sum(plane0, axis=1)
            p1 = v1 * m1 * self.dx
            m2 = np.count_nonzero(p1)
            spatial_average2 = np.sum(p1) * m2 * self.dx

        elif ((orientation.lower() == 'xz') or (orientation.lower() == 'zx')):
            nx, nz = np.shape(plane)
            x = np.linspace(0, (float(nx) - 1.0), self.Nx)
            z = np.linspace(0, (float(nz) - 1.0), self.Nz)
            dA = self.dx * self.dz
            spatial_average = simpson(simpson(plane0, z), x)
        elif ((orientation.lower() == 'zy') or (orientation.lower() == 'yz')):
            ny, nz = np.shape(plane)
            y = np.linspace(0, (float(ny) - 1.0), self.Ny)
            z = np.linspace(0, (float(nz) - 1.0), self.Nz)
            dA = self.dz * self.dy
            spatial_average = simpson(simpson(plane0, z), y)
        else:
            print("orientation false")
            spatial_average = np.NaN

        print("spatial_average:", spatial_average, "\n" +
              "spatial_average:", spatial_average1 * a,
              "spatial_average:", spatial_average2,
              "int0:", np.sum(plane0.flatten()) * (float(n) * dA),
              "int0:", np.sum(plane0.flatten()) * a,)

        return spatial_average

    def get_focus(self, p):
        """Gets value of maximum pressure and the indices of the location"""
        max_pressure = np.max(p)
        mx, my, mz = np.unravel_index(np.argmax(p, axis=None), p.shape)
        return max_pressure, [mx, my, mz]

    def get_dB_mask(self, p, dB: float = -6):
        """get focal region"""
        max_pressure, _ = self.get_focus(p)
        # find -6dB focal volume
        ratio = 10**(dB / 20.0) * max_pressure
        dummy_val = 0.0
        dB_mask = np.where(p > ratio, p, dummy_val)
        totalVolume = np.count_nonzero(dB_mask) * self.dx * self.dy * self.dz
        print('\n\tApprox. volume of FWHM {vol:8.5e} [m^3]'.format(vol=totalVolume))
        return dB_mask

    def getIsoVolume(self, p, dB=-6):
        """"Returns a triangulation of a volume, warning: may not be connected or closed"""
        max_pressure, _ = self.get_focus(p)
        ratio = 10**(dB / 20.0) * max_pressure
        verts, faces, _, _ = measure.marching_cubes(p, level=ratio, spacing=[self.dx, self.dy, self.dz])
        return verts, faces

    def getFWHM(self, p, fname: str = "fwhm.vtk"):
        """"Gets volume of -6dB field"""
        verts, faces = self.getIsoVolume(p)
        # cells = [("triangle", faces)]
        # mesh = meshio.Mesh(verts, cells)
        # mesh.write(fname)

        totalArea: float = 0.0

        m: int = np.max(np.shape(faces)) - 1
        for i in np.arange(0, m, dtype=int):
            p0 = verts[faces[m, 0]]
            p1 = verts[faces[m, 1]]
            p2 = verts[faces[m, 2]]

            a = np.asarray(p1 - p0)
            b = np.asarray(p2 - p0)

            n = np.cross(a, b)
            nn = np.abs(n)

            area = nn / 2.0
            normal = n / nn
            centre = (p0 + p1 + p2) / 3.0

            totalArea += area * (centre[0] * normal[0] + centre[1] * normal[1] + centre[2] * normal[2])

        d13 = [[verts[faces[:, 1], 0] - verts[faces[:, 2], 0]],
               [verts[faces[:, 1], 1] - verts[faces[:, 2], 1]],
               [verts[faces[:, 1], 2] - verts[faces[:, 2], 2]] ]

        d12 = [[verts[faces[:, 0], 0] - verts[faces[:, 1], 0]],
               [verts[faces[:, 0], 1] - verts[faces[:, 1], 1]],
               [verts[faces[:, 0], 2] - verts[faces[:, 1], 2]] ]

        # cross-product vectorized
        cr = np.cross(np.squeeze(np.transpose(d13)), np.squeeze(np.transpose(d12)))
        cr = np.transpose(cr)
        # Area of each triangle
        area = 0.5 * np.sqrt( cr[0, :]**2 + cr[1, :]**2 + cr[2, :]**2 )
        # Total area
        totalArea = np.sum(area)
        # norm of cross product
        crNorm = np.sqrt( cr[0, :]**2 + cr[1, :]**2 + cr[2, :]**2 )
        # centroid
        zMean = (verts[faces[:, 0], 2] + verts[faces[:, 1], 2] + verts[faces[:, 2], 2]) / 3.0
        # z component of normal for each triangle
        nz = -cr[2, :] / crNorm
        # contribution of each triangle
        volume = np.abs( np.multiply(np.multiply(area, zMean), nz) )
        # divergence theorem
        totalVolume = np.sum(volume)
        # display volume to screen
        print('\n\tTotal volume of FWHM {vol:8.5e} [m^3]'.format(vol=totalVolume))

        return verts, faces

    def outer_ellipsoid(self, verts, tol: float = 0.005, itmax: int = 500):
        """
        Find the minimum volume ellipsoid enclosing (outside) a set of points.
        Return A, c where the equation for the ellipse given in "center form" is
        (x-c).T * A * (x-c) = 1
        """

        points = np.asmatrix(verts)
        N, d = points.shape

        # print(points[:, 0:5])

        Q = np.column_stack((points, np.ones(N))).T

        u = np.ones(N, dtype=np.float64) / float(N)

        err: float = 1.0 + tol
        iter: int = 0

        convergence = []
        while ((err > tol) and (iter < itmax)):
            X = Q * np.diag(u) * Q.T
            M = np.diag(Q.T * np.linalg.inv(X) * Q)
            jdx = np.argmax(M)
            step_size = (M[jdx] - float(d) - 1.0) / ((float(d) + 1) * (M[jdx] - 1.0))
            new_u = (1.0 - step_size) * u
            new_u[jdx] += step_size
            err = np.linalg.norm(new_u - u, ord=2)
            convergence.append(err)
            iter = iter + int(1)
            u = new_u

        if (iter == itmax):
            msg = f"Maximum number of iterations reached: {iter} and error {err} greater than tolerance {tol}"
            print(msg)

        # center of ellipsoid
        c = u * points
        A = np.linalg.inv(points.T * np.diag(u) * points - c.T * c) / float(d)

        # print(c, A)

        volume = np.pi * np.sqrt(np.linalg.det(A))
        print('\n\tVolume of ellipsoid FWHM {vol:8.5e} [m^3]'.format(vol=volume))

        return np.asarray(A), np.squeeze(np.asarray(c)), np.array(convergence)

    def local_maxima_3D(self, p, n=2, order=1):
        """Detects local maxima in a 3D array

        Parameters
        ---------
        data : 3d ndarray
        n : int
            Maximum number of local maxima to find
        order : int
            How many points on each side to use for the comparison

        Returns
        -------
        coords : ndarray
            coordinates of the local maxima
        values : ndarray
            values of the local maxima
        """

        from skimage import feature

        coords = feature.peak_local_max(p, num_peaks=n)
        # coords = np.squeeze(np.asarray(coords, dtype=int))
        # print(coords, np.shape(coords), np.squeeze(np.asarray(coords, dtype=int)))
        # [print("ind:", ind) for ind in coords]
        # print(coords[0], np.array(coords[0]))
        # d = [ [[i] for i in entry] for entry in coords]
        values = []
        for i in np.arange(len(coords)):
            ind = [[i] for i in coords[i]]
            # values.append( np.asscalar(p[ind]) )
            values.append( p[ind].item() )
        # print(p[coords[0]])

        # print(p[np.array(coords[0])])

        # print(p[0,0,0])
        # print(values)

        # from scipy import ndimage as ndi
        # size = 1 + 2 * order
        # footprint = np.ones((size, size, size))
        # footprint[order, order, order] = 0

        # filtered = ndi.maximum_filter(data, footprint=footprint)
        # mask_local_maxima = data > filtered
        # coords = np.asarray(np.where(mask_local_maxima)).T
        # values = data[mask_local_maxima]

        return coords, values

    def getPVImageData(self, p, order='F'):
        pv_grid = pv.ImageData()
        pv_grid.dimensions = (self.Nx, self.Ny, self.Nz)
        pv_grid.origin = (0, 0, 0)
        pv_grid.spacing = (self.dx, self.dy, self.dz)
        pv_grid.point_data["pressure"] = p.flatten(order=order)
        pv_grid.deep_copy = False
        return pv_grid

    # def getPVUnstructuredData(self, p, order='F'):
    #     cells = [4, 0, 1, 2, 3]
    #     celltypes = [pyvista.CellType.TETRA]
    #     points = [
    #         [1.0, 1.0, 1.0],
    #         [1.0, -1.0, -1.0],
    #         [-1.0, 1.0, -1.0],
    #         [-1.0, -1.0, 1.0],
    #     pv_grid = pyvista.UnstructuredGrid(cells, celltypes, points)
    #     return pv_grid

    def plot2D(self, p, tx_plane_coords=None, verbose=False):
        """Plots 2D axial data using matplotlib"""
        from matplotlib import pyplot as plt
        from scipy.signal import find_peaks
        import seaborn as sns
        # get beam axis
        beam_axis = p[self.Ny // 2, self.Nz // 2, :]
        # scale to MPa
        beam_axis = beam_axis * 1e-6
        # get maximum pressure
        pmax, loc_pmax = self.get_focus(p)
        max_pressure = pmax * 1e-6
        max_loc = loc_pmax[2]
        dB = -6
        ratio = 10**(dB / 20.0) * max_pressure

        x_vec = np.linspace(0.0, self.dx * (self.Nx - 1.0), self.Nx)

        lower_offset = 0
        upper_offset = max_loc

        peaks, _ = find_peaks(beam_axis[lower_offset:max_loc], height=0)
        isecondary = peaks[np.argmax(beam_axis[peaks])]

        all_peaks, _ = find_peaks(beam_axis, height=0)

        new_range = all_peaks[np.argmax(beam_axis[all_peaks]) - int(1)]
        ilower = new_range + np.argmin(np.abs(beam_axis[new_range:max_loc] - ratio))

        iupper = upper_offset + np.argmin(np.abs(beam_axis[upper_offset:-1] - ratio))

        width = x_vec[iupper] - x_vec[ilower]
        if verbose:
            print("FWHM: {val:8.2e}".format(val=width))

        _, ax1 = plt.subplots()
        ax1.plot(x_vec, beam_axis,
                 color='red',
                 marker='',
                 linestyle='-',
                 linewidth=2,
                 markersize=0,
                 label='python')
        ax1.scatter(x_vec[isecondary], beam_axis[isecondary], marker='o')
        ax1.scatter([x_vec[ilower], x_vec[iupper]], [beam_axis[ilower], beam_axis[iupper]], marker='o')
        ax1.scatter([x_vec[max_loc]], [beam_axis[max_loc]], marker='o')
        ax1.fill_between(x=x_vec[ilower:iupper + 1], y1=beam_axis[ilower:iupper + 1],
                         linestyle='-', linewidth=1, facecolor='b', edgecolor='b', alpha=0.5)
        if tx_plane_coords is not None:
            ax1.vlines(x_vec[tx_plane_coords[0]], 0.0, max_pressure, linestyle='dashed')
        ax1.set(xlabel='Axial Position [m]',
                ylabel='Pressure [MPa]',
                title='Axial Pressure')
        ax1.legend()
        ax1.grid(True)

        sns.set_theme(style="darkgrid")
        _, ax2 = plt.subplots()
        ax2.plot(x_vec, beam_axis)
        ax2.scatter(x_vec[isecondary], beam_axis[isecondary], marker='*', c='b')
        ax2.scatter([x_vec[ilower], x_vec[iupper]], [beam_axis[ilower], beam_axis[iupper]], marker='o', c='b')
        ax2.scatter([x_vec[max_loc]], [beam_axis[max_loc]], marker='o', c='b')
        ax2.fill_between(x=x_vec[ilower:iupper + 1], y1=beam_axis[ilower:iupper + 1],
                         linestyle='-', linewidth=1, facecolor='b', edgecolor='b', alpha=0.5)
        if tx_plane_coords is not None:
            ax2.vlines(x_vec[tx_plane_coords[0]], 0.0, max_pressure, linestyle='dashed', colors='b')
        ax2.set(xlabel='Axial Position [m]',
                ylabel='Pressure [MPa]',
                title='Axial Pressure')

        plt.show()

    def plot3D(self, p, tx_plane_coords=None, verbose=False):
        """Plots using pyvista"""

        max_pressure, max_loc = self.get_focus(p)
        if verbose:
            print(max_pressure, max_loc)

        min_pressure = np.min(p)
        if verbose:
            print(min_pressure)

        pv_grid = self.getPVImageData(p)
        if verbose:
            print(pv_grid)

        verts, faces = self.getFWHM(p)

        cells = [("triangle", faces)]
        mesh = meshio.Mesh(verts, cells)
        mesh.write("foo2.vtk")
        dataset = pyvista.read('foo2.vtk')

        pv_x = np.linspace(0, (self.Nx - 1.0) * self.dx, self.Nx)
        pv_y = np.linspace(0, (self.Ny - 1.0) * self.dy, self.Ny)
        pv_z = np.linspace(0, (self.Nz - 1.0) * self.dz, self.Nz)

        islands = dataset.connectivity(largest=False)
        split_islands = islands.split_bodies(label=True)
        region = []
        xx = []
        for i, body in enumerate(split_islands):
            region.append(body)
            pntdata = body.GetPoints()
            xx.append(np.zeros((pntdata.GetNumberOfPoints(), 3)))
            for j in range(pntdata.GetNumberOfPoints()):
                xx[i][j, 0] = pntdata.GetPoint(j)[0]
                xx[i][j, 1] = pntdata.GetPoint(j)[1]
                xx[i][j, 2] = pntdata.GetPoint(j)[2]

        mx, my, mz = max_loc
        max_loc = [pv_x[mx], pv_y[my], pv_z[mz]]

        single_slice_x = pv_grid.slice(origin=max_loc, normal=[1, 0, 0])
        single_slice_y = pv_grid.slice(origin=max_loc, normal=[0, 1, 0])
        single_slice_z = pv_grid.slice(origin=max_loc, normal=[0, 0, 1])

        # formatting of colorbar
        sargs = dict(title='Pressure [Pa]',
                     height=0.90,
                     vertical=True,
                     position_x=0.90,
                     position_y=0.05,
                     title_font_size=20,
                     label_font_size=16,
                     shadow=False,
                     n_labels=6,
                     italic=False,
                     fmt="%.1e",
                     font_family="arial")

        # dictionary for annotations of colorbar
        ratio = 10**(-6 / 20.0) * max_pressure
        annotations = {ratio: "-6 dB"}

        # plotter object
        plotter = pyvista.Plotter()

        # slice data
        _ = plotter.add_mesh(single_slice_x,
                             cmap='turbo',
                             clim=[min_pressure, max_pressure],
                             opacity=0.5,
                             scalar_bar_args=sargs,
                             annotations=annotations)
        _ = plotter.add_mesh(single_slice_y, cmap='turbo', clim=[min_pressure, max_pressure], opacity=0.5, show_scalar_bar=False)
        _ = plotter.add_mesh(single_slice_z, cmap='turbo', clim=[min_pressure, max_pressure], opacity=0.5, show_scalar_bar=False)

        # transducer plane
        if tx_plane_coords is not None:
            tx_plane = [pv_x[tx_plane_coords[0]],
                        pv_y[tx_plane_coords[1]],
                        pv_z[tx_plane_coords[2]]]
            single_slice_tx = pv_grid.slice(origin=tx_plane, normal=[1, 0, 0])
            _ = plotter.add_mesh(single_slice_tx, cmap='turbo', clim=[min_pressure, max_pressure], opacity=0.5, show_scalar_bar=False)

        # full width half maximum
        _ = plotter.add_mesh(region[0], color='red', opacity=0.75, label='-6 dB')

        # add the frame around the image
        _ = plotter.show_bounds(grid='front',
                                location='outer',
                                ticks='outside',
                                color='black',
                                minor_ticks=False,
                                padding=0.0,
                                show_xaxis=True, show_xlabels=True, xtitle='', n_xlabels=5,
                                ytitle="",
                                ztitle="")

        # _ = plotter.add_axes(color='pink', labels_off=False)
        plotter.camera_position = 'yz'
        # plotter.camera.elevation = 45
        plotter.camera.roll = 0
        plotter.camera.azimuth = 125
        plotter.camera.elevation = 5

        # extensions = ("svg", "eps", "ps", "pdf", "tex")
        fname = "fwhm" + "." + "svg"
        plotter.save_graphic(fname, title="PyVista Export", raster=True, painter=True)

        plotter.show()


if __name__ == '__main__':

    # filename = '../../../Desktop/converted_output.h5'
    filename = "data/water/brics_water_output_1.h5"
    freq = 500e3  # must be in Hz

    data = PostProcess(filename, freq)
    sensor_data = data.hdf5_loader(filename)

    tx_plane_coords = [21, data.Ny // 2, data.Nz // 2]

    p = data.get_temporal_average(sensor_data)
    # pmax, loc_pmax = data.get_focus(p)

    data.plot2D(p, tx_plane_coords)

    # coords, values = data.local_maxima_3D(p, n=2)

    # dB = -26
    # val = data.get_spatial_average(p[:, :, loc_pmax[2]], 10**(dB / 20.0) * pmax, plot=True)

    #data.plot3D(p, tx_plane_coords)

    # # pulse length [s]
    # pulse_length = 20e-3

    # # pulse repetition frequency [Hz]
    # pulse_rep_freq = 5.0

    # # spatial peak-pulse average of plane wave intensity
    # Isppa = max_pressure**2 / (2 * medium.density * medium.sound_speed)  # [W/m2]
    # Isppa = np.squeeze(Isppa) * 1e-4  # [W/cm2]

    # # spatial peak-temporal average of plane wave intensity
    # Ispta = Isppa * pulse_length * pulse_rep_freq  # [W/cm2]

    # # Mechanical Index (MI):  max_pressure [MPa] / sqrt freq [MHz]
    # MI = max_pressure * 1e-6 / np.sqrt(freq * 1e-6)
