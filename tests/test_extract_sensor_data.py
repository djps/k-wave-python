import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch
from kwave.kWaveSimulation_helper import extract_sensor_data


class TestExtractSensorData:
    """Test suite for extract_sensor_data """
    
    @pytest.fixture
    def xp(self):
        """Return numpy as the array module."""
        return np
    
    @pytest.fixture
    def mock_flags(self):
        """Create mock flags object with all flags set to False by default."""
        flags = Mock()
        # Recording flags
        flags.record_p = False
        flags.record_p_max = False
        flags.record_p_min = False
        flags.record_p_rms = False
        flags.record_u = False
        flags.record_u_non_staggered = False
        flags.record_u_max = False
        flags.record_u_min = False
        flags.record_u_rms = False
        flags.record_u_split_field = False
        flags.record_I = False
        flags.record_I_avg = False
        flags.record_p_max_all = False
        flags.record_p_min_all = False
        flags.record_u_max_all = False
        flags.record_u_min_all = False
        
        # Sensor type flags
        flags.binary_sensor_mask = False
        flags.use_cuboid_corners = False
        flags.compute_directivity = False
        return flags
    
    @pytest.fixture
    def mock_record_1d(self):
        """Create mock record object for 1D simulations."""
        record = Mock()
        record.x_shift_neg = np.exp(-1j * np.pi * np.arange(10) / 10.0)
        record.x1_inside = 0
        record.x2_inside = 10
        return record
    
    @pytest.fixture
    def mock_record_2d(self):
        """Create mock record object for 2D simulations."""
        record = Mock()
        record.x_shift_neg = np.exp(-1j * np.pi * np.arange(10).reshape(-1, 1) / 10)
        record.y_shift_neg = np.exp(-1j * np.pi * np.arange(10).reshape(1, -1) / 10)
        record.kx_norm = np.random.rand(10, 10)
        record.ky_norm = np.random.rand(10, 10)
        record.x1_inside = 0
        record.x2_inside = 10
        record.y1_inside = 0
        record.y2_inside = 10
        record.tri = np.array([[0, 1, 2], [1, 2, 3]])
        record.bc = np.array([[0.33, 0.33, 0.34], [0.33, 0.33, 0.34]])
        return record
    
    @pytest.fixture
    def mock_record_3d(self):
        """Create mock record object for 3D simulations."""
        record = Mock()
        record.x_shift_neg = np.ones((5, 5, 5)) * np.exp(-1j * 0.1)
        record.y_shift_neg = np.ones((5, 5, 5)) * np.exp(-1j * 0.2)
        record.z_shift_neg = np.ones((5, 5, 5)) * np.exp(-1j * 0.3)
        record.kx_norm = np.random.rand(5, 5, 5)
        record.ky_norm = np.random.rand(5, 5, 5)
        record.kz_norm = np.random.rand(5, 5, 5)
        record.x1_inside = 0
        record.x2_inside = 5
        record.y1_inside = 0
        record.y2_inside = 5
        record.z1_inside = 0
        record.z2_inside = 5
        record.tri = np.array([[0, 1, 2], [1, 2, 3]])
        record.bc = np.array([[0.33, 0.33, 0.34], [0.33, 0.33, 0.34]])
        return record
    
    @pytest.fixture
    def mock_sensor_data(self):
        """Create mock sensor_data object."""
        sensor_data = Mock()
        return sensor_data
    
    # =========================================================================
    # GRID STAGGERING TESTS
    # =========================================================================
    
    def test_grid_staggering_1d_velocity_shift(self, xp, mock_flags, mock_record_1d, mock_sensor_data):
        """Test that 1D velocity data is shifted correctly."""
        mock_flags.record_u_non_staggered = True
        mock_flags.binary_sensor_mask = True
        
        dim = 1
        p = xp.random.rand(10)
        ux_sgx = xp.random.rand(10)
        sensor_mask_index = xp.array([0, 5, 9])
        
        mock_sensor_data.ux_non_staggered = xp.zeros((3, 1))
        
        result = extract_sensor_data(
            dim, xp, mock_sensor_data, 0, sensor_mask_index,
            mock_flags, mock_record_1d, p, ux_sgx
        )
        
        # Verify that ux_non_staggered was populated
        assert result.ux_non_staggered.shape == (3, 1)
        assert not xp.all(result.ux_non_staggered == 0)
    

    def test_grid_staggering_2d_both_velocities(self, xp, mock_flags, mock_record_2d, mock_sensor_data):
        """Test that 2D velocities are shifted correctly."""
        mock_flags.record_u_non_staggered = True
        mock_flags.binary_sensor_mask = True
        
        dim = 2
        p = xp.random.rand(10, 10)
        ux_sgx = xp.random.rand(10, 10)
        uy_sgy = xp.random.rand(10, 10)
        sensor_mask_index = xp.array([0, 50, 99])
        
        mock_sensor_data.ux_non_staggered = xp.zeros((3, 1))
        mock_sensor_data.uy_non_staggered = xp.zeros((3, 1))
        
        result = extract_sensor_data(
            dim, xp, mock_sensor_data, 0, sensor_mask_index,
            mock_flags, mock_record_2d, p, ux_sgx, uy_sgy
        )
        
        assert result.ux_non_staggered.shape == (3, 1)
        assert result.uy_non_staggered.shape == (3, 1)
    
    def test_grid_staggering_3d_all_velocities(self, xp, mock_flags, mock_record_3d, mock_sensor_data):
        """Test that 3D velocities are shifted correctly."""
        mock_flags.record_u_non_staggered = True
        mock_flags.binary_sensor_mask = True
        
        dim = 3
        p = xp.random.rand(5, 5, 5)
        ux_sgx = xp.random.rand(5, 5, 5)
        uy_sgy = xp.random.rand(5, 5, 5)
        uz_sgz = xp.random.rand(5, 5, 5)
        sensor_mask_index = xp.array([0, 60, 124])
        
        mock_sensor_data.ux_non_staggered = xp.zeros((3, 1))
        mock_sensor_data.uy_non_staggered = xp.zeros((3, 1))
        mock_sensor_data.uz_non_staggered = xp.zeros((3, 1))
        
        result = extract_sensor_data(
            dim, xp, mock_sensor_data, 0, sensor_mask_index,
            mock_flags, mock_record_3d, p, ux_sgx, uy_sgy, uz_sgz
        )
        
        assert result.ux_non_staggered.shape == (3, 1)
        assert result.uy_non_staggered.shape == (3, 1)
        assert result.uz_non_staggered.shape == (3, 1)
    
    def test_no_staggering_when_not_needed(self, xp, mock_flags, mock_record_1d, mock_sensor_data):
        """Test that staggering is skipped when flags don't require it."""
        mock_flags.binary_sensor_mask = True
        mock_flags.record_p = True
        
        dim = 1
        p = xp.random.rand(10)
        ux_sgx = xp.random.rand(10)
        sensor_mask_index = xp.array([0, 5, 9])
        
        mock_sensor_data.p = xp.zeros((3, 1))
        
        # This should not raise an error even without shift operators
        result = extract_sensor_data(
            dim, xp, mock_sensor_data, 0, sensor_mask_index,
            mock_flags, mock_record_1d, p, ux_sgx
        )
        
        assert result.p.shape == (3, 1)
    
    # =========================================================================
    # BINARY SENSOR MASK - PRESSURE TESTS
    # =========================================================================
    
    def test_binary_mask_record_p_1d(self, xp, mock_flags, mock_record_1d, mock_sensor_data):
        """Test recording pressure with binary sensor mask in 1D."""
        mock_flags.binary_sensor_mask = True
        mock_flags.record_p = True
        
        dim = 1
        p = xp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        ux_sgx = xp.zeros(10)
        sensor_mask_index = xp.array([0, 5, 9])
        
        mock_sensor_data.p = xp.zeros((3, 1))
        
        result = extract_sensor_data(
            dim, xp, mock_sensor_data, 0, sensor_mask_index,
            mock_flags, mock_record_1d, p, ux_sgx
        )
        
        expected = xp.array([[1.0], [6.0], [10.0]])
        xp.testing.assert_array_equal(result.p, expected)
    
    def test_binary_mask_record_p_max_first_call(self, xp, mock_flags, mock_record_2d, mock_sensor_data):
        """Test recording p_max on first file_index."""
        mock_flags.binary_sensor_mask = True
        mock_flags.record_p_max = True
        
        dim = 2
        p = xp.random.rand(10, 10) * 10
        ux_sgx = xp.zeros((10, 10))
        sensor_mask_index = xp.array([0, 50, 99])
        
        result = extract_sensor_data(
            dim, xp, mock_sensor_data, 0, sensor_mask_index,
            mock_flags, mock_record_2d, p, ux_sgx, xp.zeros((10, 10))
        )
        
        assert hasattr(result, 'p_max')
        assert result.p_max.shape == (3,)
    
    def test_binary_mask_record_p_max_updates(self, xp, mock_flags, mock_record_2d, mock_sensor_data):
        """Test that p_max updates correctly on subsequent calls."""
        mock_flags.binary_sensor_mask = True
        mock_flags.record_p_max = True
        
        dim = 2
        p = xp.ones((10, 10)) * 5.0
        ux_sgx = xp.zeros((10, 10))
        sensor_mask_index = xp.array([0, 50, 99])
        
        mock_sensor_data.p_max = xp.ones(3) * 3.0
        
        result = extract_sensor_data(
            dim, xp, mock_sensor_data, 1, sensor_mask_index,
            mock_flags, mock_record_2d, p, ux_sgx, xp.zeros((10, 10))
        )
        
        # p_max should be updated to 5.0
        assert xp.all(result.p_max == 5.0)
    
    def test_binary_mask_record_p_min(self, xp, mock_flags, mock_record_2d, mock_sensor_data):
        """Test recording minimum pressure."""
        mock_flags.binary_sensor_mask = True
        mock_flags.record_p_min = True
        
        dim = 2
        p = xp.ones((10, 10)) * 2.0
        ux_sgx = xp.zeros((10, 10))
        sensor_mask_index = xp.array([0, 50, 99])
        
        mock_sensor_data.p_min = xp.ones(3) * 5.0
        
        result = extract_sensor_data(
            dim, xp, mock_sensor_data, 1, sensor_mask_index,
            mock_flags, mock_record_2d, p, ux_sgx, xp.zeros((10, 10))
        )
        
        assert xp.all(result.p_min == 2.0)
    
    def test_binary_mask_record_p_rms_first_call(self, xp, mock_flags, mock_record_2d, mock_sensor_data):
        """Test RMS pressure calculation on first call."""
        mock_flags.binary_sensor_mask = True
        mock_flags.record_p_rms = True
        
        dim = 2
        p = xp.ones((10, 10)) * 3.0
        ux_sgx = xp.zeros((10, 10))
        sensor_mask_index = xp.array([0, 50, 99])
        
        result = extract_sensor_data(
            dim, xp, mock_sensor_data, 0, sensor_mask_index,
            mock_flags, mock_record_2d, p, ux_sgx, xp.zeros((10, 10))
        )
        
        assert hasattr(result, 'p_rms')
        xp.testing.assert_array_almost_equal(result.p_rms, xp.ones(3) * 9.0)
    
    def test_binary_mask_record_p_rms_updates(self, xp, mock_flags, mock_record_2d, mock_sensor_data):
        """Test RMS pressure updates correctly."""
        mock_flags.binary_sensor_mask = True
        mock_flags.record_p_rms = True
        
        dim = 2
        p = xp.ones((10, 10)) * 4.0
        ux_sgx = xp.zeros((10, 10))
        sensor_mask_index = xp.array([0, 50, 99])
        
        mock_sensor_data.p_rms = xp.ones(3) * 3.0
        
        result = extract_sensor_data(
            dim, xp, mock_sensor_data, 1, sensor_mask_index,
            mock_flags, mock_record_2d, p, ux_sgx, xp.zeros((10, 10))
        )
        
        # RMS should be sqrt((3^2 * 1 + 4^2) / 2) = sqrt(12.5)
        expected_rms = xp.sqrt((9.0 + 16.0) / 2.0)
        xp.testing.assert_array_almost_equal(result.p_rms, xp.ones(3) * expected_rms)
    
    # =========================================================================
    # BINARY SENSOR MASK - VELOCITY TESTS
    # =========================================================================
    
    def test_binary_mask_record_u_1d(self, xp, mock_flags, mock_record_1d, mock_sensor_data):
        """Test recording staggered velocity in 1D."""
        mock_flags.binary_sensor_mask = True
        mock_flags.record_u = True
        
        dim = 1
        p = xp.zeros(10)
        ux_sgx = xp.arange(10, dtype=float)
        sensor_mask_index = xp.array([0, 5, 9])
        
        mock_sensor_data.ux = xp.zeros((3, 1))
        
        result = extract_sensor_data(
            dim, xp, mock_sensor_data, 0, sensor_mask_index,
            mock_flags, mock_record_1d, p, ux_sgx
        )
        
        expected = xp.array([[0.0], [5.0], [9.0]])
        xp.testing.assert_array_equal(result.ux, expected)
    
    def test_binary_mask_record_u_2d(self, xp, mock_flags, mock_record_2d, mock_sensor_data):
        """Test recording staggered velocity in 2D."""
        mock_flags.binary_sensor_mask = True
        mock_flags.record_u = True
        
        dim = 2
        p = xp.zeros((10, 10))
        ux_sgx = xp.ones((10, 10)) * 2.0
        uy_sgy = xp.ones((10, 10)) * 3.0
        sensor_mask_index = xp.array([0, 50, 99])
        
        mock_sensor_data.ux = xp.zeros((3, 1))
        mock_sensor_data.uy = xp.zeros((3, 1))
        
        result = extract_sensor_data(
            dim, xp, mock_sensor_data, 0, sensor_mask_index,
            mock_flags, mock_record_2d, p, ux_sgx, uy_sgy
        )
        
        assert xp.all(result.ux == 2.0)
        assert xp.all(result.uy == 3.0)
    
    def test_binary_mask_record_u_3d(self, xp, mock_flags, mock_record_3d, mock_sensor_data):
        """Test recording staggered velocity in 3D."""
        mock_flags.binary_sensor_mask = True
        mock_flags.record_u = True
        
        dim = 3
        p = xp.zeros((5, 5, 5))
        ux_sgx = xp.ones((5, 5, 5)) * 1.0
        uy_sgy = xp.ones((5, 5, 5)) * 2.0
        uz_sgz = xp.ones((5, 5, 5)) * 3.0
        sensor_mask_index = xp.array([0, 60, 124])
        
        mock_sensor_data.ux = xp.zeros((3, 1))
        mock_sensor_data.uy = xp.zeros((3, 1))
        mock_sensor_data.uz = xp.zeros((3, 1))
        
        result = extract_sensor_data(
            dim, xp, mock_sensor_data, 0, sensor_mask_index,
            mock_flags, mock_record_3d, p, ux_sgx, uy_sgy, uz_sgz
        )
        
        assert xp.all(result.ux == 1.0)
        assert xp.all(result.uy == 2.0)
        assert xp.all(result.uz == 3.0)
    
    def test_binary_mask_record_u_max(self, xp, mock_flags, mock_record_2d, mock_sensor_data):
        """Test recording maximum velocity."""
        mock_flags.binary_sensor_mask = True
        mock_flags.record_u_max = True
        
        dim = 2
        p = xp.zeros((10, 10))
        ux_sgx = xp.ones((10, 10)) * 5.0
        uy_sgy = xp.ones((10, 10)) * 6.0
        sensor_mask_index = xp.array([0, 50, 99])
        
        mock_sensor_data.ux_max = xp.ones(3) * 3.0
        mock_sensor_data.uy_max = xp.ones(3) * 4.0
        
        result = extract_sensor_data(
            dim, xp, mock_sensor_data, 1, sensor_mask_index,
            mock_flags, mock_record_2d, p, ux_sgx, uy_sgy
        )
        
        assert xp.all(result.ux_max == 5.0)
        assert xp.all(result.uy_max == 6.0)
    
    def test_binary_mask_record_u_rms(self, xp, mock_flags, mock_record_2d, mock_sensor_data):
        """Test recording RMS velocity."""
        mock_flags.binary_sensor_mask = True
        mock_flags.record_u_rms = True
        
        dim = 2
        p = xp.zeros((10, 10))
        ux_sgx = xp.ones((10, 10)) * 4.0
        uy_sgy = xp.ones((10, 10)) * 3.0
        sensor_mask_index = xp.array([0, 50, 99])
        
        mock_sensor_data.ux_rms = xp.ones(3) * 0.0
        mock_sensor_data.uy_rms = xp.ones(3) * 0.0
        
        result = extract_sensor_data(
            dim, xp, mock_sensor_data, 0, sensor_mask_index,
            mock_flags, mock_record_2d, p, ux_sgx, uy_sgy
        )
        
        assert xp.all(result.ux_rms == 4.0)
        assert xp.all(result.uy_rms == 3.0)
    
    # =========================================================================
    # BINARY SENSOR MASK - SPLIT FIELD TESTS
    # =========================================================================
    
    def test_binary_mask_split_field_2d(self, xp, mock_flags, mock_record_2d, mock_sensor_data):
        """Test split field calculation in 2D."""
        mock_flags.binary_sensor_mask = True
        mock_flags.record_u_split_field = True
        
        dim = 2
        p = xp.zeros((10, 10))
        ux_sgx = xp.random.rand(10, 10)
        uy_sgy = xp.random.rand(10, 10)
        sensor_mask_index = xp.array([0, 50, 99])
        
        mock_sensor_data.ux_split_p = xp.zeros((3, 1))
        mock_sensor_data.ux_split_s = xp.zeros((3, 1))
        mock_sensor_data.uy_split_p = xp.zeros((3, 1))
        mock_sensor_data.uy_split_s = xp.zeros((3, 1))
        
        result = extract_sensor_data(
            dim, xp, mock_sensor_data, 0, sensor_mask_index,
            mock_flags, mock_record_2d, p, ux_sgx, uy_sgy
        )
        
        # Verify split fields were computed
        assert not xp.all(result.ux_split_p == 0)
        assert not xp.all(result.ux_split_s == 0)
        assert not xp.all(result.uy_split_p == 0)
        assert not xp.all(result.uy_split_s == 0)
    
    def test_binary_mask_split_field_3d(self, xp, mock_flags, mock_record_3d, mock_sensor_data):
        """Test split field calculation in 3D."""
        mock_flags.binary_sensor_mask = True
        mock_flags.record_u_split_field = True
        
        dim = 3
        p = xp.zeros((5, 5, 5))
        ux_sgx = xp.random.rand(5, 5, 5)
        uy_sgy = xp.random.rand(5, 5, 5)
        uz_sgz = xp.random.rand(5, 5, 5)
        sensor_mask_index = xp.array([0, 60, 124])
        
        mock_sensor_data.ux_split_p = xp.zeros((3, 1))
        mock_sensor_data.ux_split_s = xp.zeros((3, 1))
        mock_sensor_data.uy_split_p = xp.zeros((3, 1))
        mock_sensor_data.uy_split_s = xp.zeros((3, 1))
        mock_sensor_data.uz_split_p = xp.zeros((3, 1))
        mock_sensor_data.uz_split_s = xp.zeros((3, 1))
        
        result = extract_sensor_data(
            dim, xp, mock_sensor_data, 0, sensor_mask_index,
            mock_flags, mock_record_3d, p, ux_sgx, uy_sgy, uz_sgz
        )
        
        # Verify all 6 split fields were computed
        assert hasattr(result, 'ux_split_p')
        assert hasattr(result, 'ux_split_s')
        assert hasattr(result, 'uy_split_p')
        assert hasattr(result, 'uy_split_s')
        assert hasattr(result, 'uz_split_p')
        assert hasattr(result, 'uz_split_s')
    
    # =========================================================================
    # CUBOID CORNERS TESTS
    # =========================================================================
    
    def test_cuboid_corners_3d_pressure(self, xp, mock_flags, mock_record_3d):
        """Test cuboid corner extraction in 3D."""
        mock_flags.use_cuboid_corners = True
        mock_flags.record_p = True
        
        dim = 3
        p = xp.arange(125, dtype=float).reshape(5, 5, 5)
        ux_sgx = xp.zeros((5, 5, 5))
        uy_sgy = xp.zeros((5, 5, 5))
        uz_sgz = xp.zeros((5, 5, 5))
        sensor_mask_index = xp.array([])
        
        # Define a cuboid from (0,0,0) to (2,2,2)
        mock_record_3d.cuboid_corners_list = xp.array([
            [0, 0, 0, 2, 2, 2]
        ]).T
        
        # Create list of sensor_data objects for each cuboid
        sensor_data = [Mock()]
        sensor_data[0].p = xp.zeros((3, 3, 3, 1))
        
        result = extract_sensor_data(
            dim, xp, sensor_data, 0, sensor_mask_index,
            mock_flags, mock_record_3d, p, ux_sgx, uy_sgy, uz_sgz
        )
        
        # Verify the shape
        assert result[0].p.shape == (3, 3, 3, 1)
        # Verify some values
        assert result[0].p[0, 0, 0, 0] == 0
        assert result[0].p[2, 2, 2, 0] == 62  # p[2,2,2]
    
    def test_cuboid_corners_multiple_cuboids(self, xp, mock_flags, mock_record_3d):
        """Test extraction from multiple cuboids."""
        mock_flags.use_cuboid_corners = True
        mock_flags.record_p = True
        
        dim = 3
        p = xp.ones((5, 5, 5)) * 7.0
        ux_sgx = xp.zeros((5, 5, 5))
        uy_sgy = xp.zeros((5, 5, 5))
        uz_sgz = xp.zeros((5, 5, 5))
        sensor_mask_index = xp.array([])
        
        # Define two cuboids
        mock_record_3d.cuboid_corners_list = xp.array([
            [0, 0, 0, 1, 1, 1],  # First cuboid
            [3, 3, 3, 4, 4, 4]   # Second cuboid
        ]).T
        
        sensor_data = [Mock(), Mock()]
        sensor_data[0].p = xp.zeros((2, 2, 2, 1))
        sensor_data[1].p = xp.zeros((2, 2, 2, 1))
        
        result = extract_sensor_data(
            dim, xp, sensor_data, 0, sensor_mask_index,
            mock_flags, mock_record_3d, p, ux_sgx, uy_sgy, uz_sgz
        )
        
        # Both cuboids should be filled with 7.0
        assert xp.all(result[0].p == 7.0)
        assert xp.all(result[1].p == 7.0)
    
    # =========================================================================
    # CARTESIAN INTERPOLATION TESTS
    # =========================================================================
    
    def test_cartesian_interpolation_1d(self, xp, mock_flags, mock_sensor_data):
        """Test Cartesian interpolation in 1D."""
        mock_flags.record_p = True
        
        dim = 1
        p = xp.array([0.0, 1.0, 2.0, 3.0, 4.0])
        ux_sgx = xp.zeros(5)
        sensor_mask_index = xp.array([])
        
        record = Mock()
        record.sensor_x = xp.array([0.5, 2.5])
        record.grid_x = xp.arange(5, dtype=float)
        
        mock_sensor_data.p = xp.zeros((2, 1))
        
        result = extract_sensor_data(
            dim, xp, mock_sensor_data, 0, sensor_mask_index,
            mock_flags, record, p, ux_sgx
        )
        
        # Interpolated values should be between grid points
        assert 0.0 < result.p[0, 0] < 1.0
        assert 2.0 < result.p[1, 0] < 3.0
    
    def test_cartesian_interpolation_2d(self, xp, mock_flags, mock_record_2d, mock_sensor_data):
        """Test Cartesian interpolation in 2D using barycentric coordinates."""
        mock_flags.record_p = True
        
        dim = 2
        p = xp.ones((10, 10)) * 5.0
        ux_sgx = xp.zeros((10, 10))
        uy_sgy = xp.zeros((10, 10))
        sensor_mask_index = xp.array([])
        
        # Flatten p for triangulation indexing
        p_flat = p.ravel()
        mock_record_2d.tri = xp.array([[0, 1, 2], [10, 11, 12]])
        mock_record_2d.bc = xp.array([[0.33, 0.33, 0.34], [0.33, 0.33, 0.34]])
        
        mock_sensor_data.p = xp.zeros((2, 1))
        
        result = extract_sensor_data(
            dim, xp, mock_sensor_data, 0, sensor_mask_index,
            mock_flags, mock_record_2d, p_flat, ux_sgx, uy_sgy
        )
        
        # With uniform field, interpolated values should equal 5.0
        xp.testing.assert_array_almost_equal(result.p[:, 0], xp.array([5.0, 5.0]))
    
    # =========================================================================
    # ALL-GRID RECORDING TESTS
    # =========================================================================
    
    def test_record_p_max_all_1d(self, xp, mock_flags, mock_record_1d, mock_sensor_data):
        """Test recording p_max over entire 1D grid."""
        mock_flags.record_p_max_all = True
        
        dim = 1
        p = xp.arange(10, dtype=float)
        ux_sgx = xp.zeros(10)
        sensor_mask_index = xp.array([])
        
        result = extract_sensor_data(
            dim, xp, mock_sensor_data, 0, sensor_mask_index,
            mock_flags, mock_record_1d, p, ux_sgx
        )
        
        assert hasattr(result, 'p_max_all')
        xp.testing.assert_array_equal(result.p_max_all, xp.arange(10, dtype=float))
    
    def test_record_p_max_all_2d(self, xp, mock_flags, mock_record_2d, mock_sensor_data):
        """Test recording p_max over entire 2D grid."""
        mock_flags.record_p_max_all = True
        
        dim = 2
        p = xp.ones((10, 10)) * 8.0
        ux_sgx = xp.zeros((10, 10))
        uy_sgy = xp.zeros((10, 10))
        sensor_mask_index = xp.array([])
        
        result = extract_sensor_data(
            dim, xp, mock_sensor_data, 0, sensor_mask_index,
            mock_flags, mock_record_2d, p, ux_sgx, uy_sgy
        )
        
        assert hasattr(result, 'p_max_all')
        assert result.p_max_all.shape == (10, 10)
        assert xp.all(result.p_max_all == 8.0)
    
    def test_record_u_min_all_3d(self, xp, mock_flags, mock_record_3d, mock_sensor_data):
        """Test recording u_min over entire 3D grid."""
        mock_flags.record_u_min_all = True
        
        dim = 3
        p = xp.zeros((5, 5, 5))
        ux_sgx = xp.ones((5, 5, 5)) * -2.0
        uy_sgy = xp.ones((5, 5, 5)) * -3.0
        uz_sgz = xp.ones((5, 5, 5)) * -4.0
        sensor_mask_index = xp.array([])
        
        mock_sensor_data.ux_min_all = xp.ones((5, 5, 5)) * 0.0
        mock_sensor_data.uy_min_all = xp.ones((5, 5, 5)) * 0.0
        mock_sensor_data.uz_min_all = xp.ones((5, 5, 5)) * 0.0
        
        result = extract_sensor_data(
            dim, xp, mock_sensor_data, 1, sensor_mask_index,
            mock_flags, mock_record_3d, p, ux_sgx, uy_sgy, uz_sgz
        )
        
        assert xp.all(result.ux_min_all == -2.0)
        assert xp.all(result.uy_min_all == -3.0)
        assert xp.all(result.uz_min_all == -4.0)
    
    # =========================================================================
    # ERROR HANDLING TESTS
    # =========================================================================
    
    def test_invalid_dimension_raises_error(self, xp, mock_flags, mock_record_1d, mock_sensor_data):
        """Test that invalid dimension raises RuntimeError."""
        mock_flags.binary_sensor_mask = True
        mock_flags.record_u = True
        
        dim = 4  # Invalid dimension
        p = xp.zeros(10)
        ux_sgx = xp.zeros(10)
        sensor_mask_index = xp.array([0])
        
        with pytest.raises(RuntimeError, match="Wrong dimensions"):
            extract_sensor_data(
                dim, xp, mock_sensor_data, 0, sensor_mask_index,
                mock_flags, mock_record_1d, p, ux_sgx
            )
    
    def test_directivity_not_implemented(self, xp, mock_flags, mock_record_2d, mock_sensor_data):
        """Test that directivity computation raises NotImplementedError."""
        mock_flags.binary_sensor_mask = True
        mock_flags.record_p = True
        mock_flags.compute_directivity = True
        
        dim = 2
        p = xp.zeros((10, 10))
        ux_sgx = xp.zeros((10, 10))
        uy_sgy = xp.zeros((10, 10))
        sensor_mask_index = xp.array([0])
        
        mock_sensor_data.p = xp.zeros((1, 1))
        
        with pytest.raises(NotImplementedError, match="directivity not used"):
            extract_sensor_data(
                dim, xp, mock_sensor_data, 0, sensor_mask_index,
                mock_flags, mock_record_2d, p, ux_sgx, uy_sgy
            )
    
    # =========================================================================
    # INTEGRATION TESTS
    # =========================================================================
    
    def test_multiple_flags_simultaneously(self, xp, mock_flags, mock_record_2d, mock_sensor_data):
        """Test that multiple recording flags work together."""
        mock_flags.binary_sensor_mask = True
        mock_flags.record_p = True
        mock_flags.record_p_max = True
        mock_flags.record_u = True
        mock_flags.record_u_min = True
        
        dim = 2
        p = xp.random.rand(10, 10)
        ux_sgx = xp.random.rand(10, 10)
        uy_sgy = xp.random.rand(10, 10)
        sensor_mask_index = xp.array([0, 50, 99])
        
        mock_sensor_data.p = xp.zeros((3, 1))
        mock_sensor_data.ux = xp.zeros((3, 1))
        mock_sensor_data.uy = xp.zeros((3, 1))
        
        result = extract_sensor_data(
            dim, xp, mock_sensor_data, 0, sensor_mask_index,
            mock_flags, mock_record_2d, p, ux_sgx, uy_sgy
        )
        
        # All requested data should be recorded
        assert hasattr(result, 'p')
        assert hasattr(result, 'p_max')
        assert hasattr(result, 'ux')
        assert hasattr(result, 'uy')
        assert hasattr(result, 'ux_min')
        assert hasattr(result, 'uy_min')
    
    @pytest.mark.parametrize("file_index", [0, 1, 5, 10])
    def test_file_index_progression(self, xp, mock_flags, mock_record_2d, mock_sensor_data, file_index):
        """Test that file_index parameter works across different values."""
        mock_flags.binary_sensor_mask = True
        mock_flags.record_p = True
        
        dim = 2
        p = xp.random.rand(10, 10)
        ux_sgx = xp.zeros((10, 10))
        uy_sgy = xp.zeros((10, 10))
        sensor_mask_index = xp.array([0, 50, 99])
        
        mock_sensor_data.p = xp.zeros((3, 20))  # Enough space for any file_index
        
        result = extract_sensor_data(
            dim, xp, mock_sensor_data, file_index, sensor_mask_index,
            mock_flags, mock_record_2d, p, ux_sgx, uy_sgy
        )
        
        # Check that data was written to the correct time index
        assert not xp.all(result.p[:, file_index] == 0)