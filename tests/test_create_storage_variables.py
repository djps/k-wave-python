import pytest
import numpy as np
from numpy.fft import ifftshift
from unittest.mock import Mock

from kwave.kWaveSimulation_helper.create_storage_variables import get_num_of_sensor_points, create_normalized_wavenumber_vectors


class TestGetNumOfSensorPoints:
    """Test suite for get_num_of_sensor_points function."""
    
    def test_blank_sensor(self):
        """Test when sensor is blank - should return kgrid_k.size"""
        kgrid_k = np.array([[1, 2, 3], [4, 5, 6]])
        result = get_num_of_sensor_points(
            is_blank_sensor=True,
            is_binary_sensor_mask=False,
            kgrid_k=kgrid_k,
            sensor_mask_index=[],
            sensor_x=[]
        )
        assert result == 6  # 2x3 array has size 6
    
    def test_binary_sensor_mask(self):
        """Test when binary sensor mask is used - should return len(sensor_mask_index)"""
        kgrid_k = np.array([[1, 2, 3]])
        sensor_mask_index = [0, 1, 2, 3, 4]
        result = get_num_of_sensor_points(
            is_blank_sensor=False,
            is_binary_sensor_mask=True,
            kgrid_k=kgrid_k,
            sensor_mask_index=sensor_mask_index,
            sensor_x=[]
        )
        assert result == 5
    
    def test_cartesian_sensor(self):
        """Test when Cartesian sensor is used - should return len(sensor_x)"""
        kgrid_k = np.array([1, 2, 3])
        sensor_x = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
        result = get_num_of_sensor_points(
            is_blank_sensor=False,
            is_binary_sensor_mask=False,
            kgrid_k=kgrid_k,
            sensor_mask_index=[],
            sensor_x=sensor_x
        )
        assert result == 7
    
    def test_blank_sensor_precedence(self):
        """Test that blank sensor takes precedence over other options"""
        kgrid_k = np.array([1, 2, 3, 4])
        result = get_num_of_sensor_points(
            is_blank_sensor=True,
            is_binary_sensor_mask=True,  # This should be ignored
            kgrid_k=kgrid_k,
            sensor_mask_index=[0, 1, 2],
            sensor_x=[0.1, 0.2]
        )
        assert result == 4
    
    def test_binary_mask_precedence(self):
        """Test that binary mask takes precedence over Cartesian"""
        kgrid_k = np.array([1, 2, 3])
        result = get_num_of_sensor_points(
            is_blank_sensor=False,
            is_binary_sensor_mask=True,
            kgrid_k=kgrid_k,
            sensor_mask_index=[0, 1, 2],
            sensor_x=[0.1, 0.2, 0.3, 0.4, 0.5]  # Should be ignored
        )
        assert result == 3
    
    def test_empty_sensors(self):
        """Test with empty sensor lists"""
        kgrid_k = np.array([1])
        result = get_num_of_sensor_points(
            is_blank_sensor=False,
            is_binary_sensor_mask=False,
            kgrid_k=kgrid_k,
            sensor_mask_index=[],
            sensor_x=[]
        )
        assert result == 0
    
    def test_3d_kgrid(self):
        """Test with 3D kgrid array"""
        kgrid_k = np.random.rand(10, 20, 30)
        result = get_num_of_sensor_points(
            is_blank_sensor=True,
            is_binary_sensor_mask=False,
            kgrid_k=kgrid_k,
            sensor_mask_index=[],
            sensor_x=[]
        )
        assert result == 6000  # 10 * 20 * 30
    
    @pytest.mark.parametrize("num_points", [1, 5, 10, 50, 100])
    def test_parametrized_sensor_mask_lengths(self, num_points):
        """Test with various sensor_mask_index lengths"""
        kgrid_k = np.array([1, 2, 3])
        sensor_mask_index = list(range(num_points))
        result = get_num_of_sensor_points(
            is_blank_sensor=False,
            is_binary_sensor_mask=True,
            kgrid_k=kgrid_k,
            sensor_mask_index=sensor_mask_index,
            sensor_x=[]
        )
        assert result == num_points




class TestCreateNormalizedWavenumberVectors:
    """Test suite for create_normalized_wavenumber_vectors function."""
    
    @pytest.fixture
    def mock_record(self):
        """Create a mock Recorder object."""
        return Mock()
    
    @pytest.fixture
    def mock_kgrid_1d(self):
        """Create a mock 1D kWaveGrid object."""
        kgrid = Mock()
        kgrid.dim = 1
        kgrid.kx = np.array([0, 1, 2, 3, 4])
        kgrid.k = np.array([0, 1, 2, 3, 4])
        return kgrid
    
    @pytest.fixture
    def mock_kgrid_2d(self):
        """Create a mock 2D kWaveGrid object."""
        kgrid = Mock()
        kgrid.dim = 2
        kgrid.kx = np.array([[0, 1, 2], [3, 4, 5]])
        kgrid.ky = np.array([[1, 2, 3], [4, 5, 6]])
        kgrid.k = np.array([[1, 2, 3], [4, 5, 6]])
        return kgrid
    
    @pytest.fixture
    def mock_kgrid_3d(self):
        """Create a mock 3D kWaveGrid object."""
        kgrid = Mock()
        kgrid.dim = 3
        kgrid.kx = np.random.rand(3, 4, 5)
        kgrid.ky = np.random.rand(3, 4, 5)
        kgrid.kz = np.random.rand(3, 4, 5)
        kgrid.k = np.random.rand(3, 4, 5) + 0.1  # Avoid zeros
        return kgrid
    


    def test_no_split_field_returns_unchanged(self, mock_record, mock_kgrid_1d):
        """Test that record is returned unchanged when is_record_u_split_field is False."""
        result = create_normalized_wavenumber_vectors(
            mock_record, 
            mock_kgrid_1d, 
            is_record_u_split_field=False
        )
        
        assert result is mock_record
        assert not hasattr(result, 'kx_norm'), "kx_norm should not be created when is_record_u_split_field is False"


    
    def test_1d_creates_kx_norm(self, mock_record, mock_kgrid_1d):
        """Test that kx_norm is created for 1D grid."""
        result = create_normalized_wavenumber_vectors(
            mock_record, 
            mock_kgrid_1d, 
            is_record_u_split_field=True
        )
        
        assert hasattr(result, 'kx_norm')
        assert result.kx_norm.shape == mock_kgrid_1d.kx.shape
    
    def test_1d_normalization_correctness(self, mock_record, mock_kgrid_1d):
        """Test that kx_norm is correctly normalized in 1D."""
        result = create_normalized_wavenumber_vectors(
            mock_record, 
            mock_kgrid_1d, 
            is_record_u_split_field=True
        )
        
        # Manual calculation for verification
        expected = mock_kgrid_1d.kx / mock_kgrid_1d.k
        expected[mock_kgrid_1d.k == 0] = 0
        expected = np.fft.ifftshift(expected)
        
        np.testing.assert_array_almost_equal(result.kx_norm, expected)
    

    def test_1d_handles_zero_wavenumber(self, mock_record):
        """Test that division by zero is handled correctly in 1D."""
        kgrid = Mock()
        kgrid.dim = 1
        kgrid.kx = np.array([0, 1, 2, 0, 4])
        kgrid.k = np.array([0, 1, 2, 0, 4])
        
        result = create_normalized_wavenumber_vectors(
            mock_record, 
            kgrid, 
            is_record_u_split_field=True
        )
        
        # Check that zeros in k result in zeros in kx_norm
        assert result.kx_norm[0] == 0, f"kx_norm should be 0 where k is 0, got {result.kx_norm[0]}"
        assert result.kx_norm[3] == 0, f"kx_norm should be 0 where k is 0, got {result.kx_norm[3]}"
    

    def test_2d_creates_both_norms(self, mock_record, mock_kgrid_2d):
        """Test that both kx_norm and ky_norm are created for 2D grid."""
        result = create_normalized_wavenumber_vectors(
            mock_record, 
            mock_kgrid_2d, 
            is_record_u_split_field=True
        )
        
        assert hasattr(result, 'kx_norm')
        assert hasattr(result, 'ky_norm')
        assert result.kx_norm.shape == mock_kgrid_2d.kx.shape
        assert result.ky_norm.shape == mock_kgrid_2d.ky.shape
    

    def test_2d_normalization_correctness(self, mock_record, mock_kgrid_2d):
        """Test that normalization is correct for 2D grid."""
        result = create_normalized_wavenumber_vectors(
            mock_record, 
            mock_kgrid_2d, 
            is_record_u_split_field=True
        )
        
        # Manual calculation for kx
        expected_kx = mock_kgrid_2d.kx / mock_kgrid_2d.k
        expected_kx[mock_kgrid_2d.k == 0] = 0
        expected_kx = ifftshift(expected_kx)
        
        # Manual calculation for ky
        expected_ky = mock_kgrid_2d.ky / mock_kgrid_2d.k
        expected_ky[mock_kgrid_2d.k == 0] = 0
        expected_ky = ifftshift(expected_ky)
        
        np.testing.assert_array_almost_equal(result.kx_norm, expected_kx)
        np.testing.assert_array_almost_equal(result.ky_norm, expected_ky)
    

    def test_2d_handles_zero_wavenumber(self, mock_record):
        """Test that division by zero is handled correctly in 2D."""
        kgrid = Mock()
        kgrid.dim = 2
        kgrid.kx = np.array([[0, 1], [2, 0]])
        kgrid.ky = np.array([[1, 2], [0, 3]])
        kgrid.k = np.array([[0, 2], [0, 3]])
        
        result = create_normalized_wavenumber_vectors(
            mock_record, 
            kgrid, 
            is_record_u_split_field=True
        )
        
        # After ifftshift, check that zeros are preserved
        assert not np.isnan(result.kx_norm).any()
        assert not np.isnan(result.ky_norm).any()
    

    def test_3d_creates_all_norms(self, mock_record, mock_kgrid_3d):
        """Test that kx_norm, ky_norm, and kz_norm are created for 3D grid."""
        result = create_normalized_wavenumber_vectors(
            mock_record, 
            mock_kgrid_3d, 
            is_record_u_split_field=True
        )
        
        assert hasattr(result, 'kx_norm')
        assert hasattr(result, 'ky_norm')
        assert hasattr(result, 'kz_norm')
        assert result.kx_norm.shape == mock_kgrid_3d.kx.shape
        assert result.ky_norm.shape == mock_kgrid_3d.ky.shape
        assert result.kz_norm.shape == mock_kgrid_3d.kz.shape
    

    def test_3d_normalization_correctness(self, mock_record, mock_kgrid_3d):
        """Test that normalization is correct for 3D grid."""
        result = create_normalized_wavenumber_vectors(
            mock_record, 
            mock_kgrid_3d, 
            is_record_u_split_field=True
        )
        
        # Manual calculation for kx
        expected_kx = mock_kgrid_3d.kx / mock_kgrid_3d.k
        expected_kx[mock_kgrid_3d.k == 0] = 0
        expected_kx = ifftshift(expected_kx)
        
        # Manual calculation for ky
        expected_ky = mock_kgrid_3d.ky / mock_kgrid_3d.k
        expected_ky[mock_kgrid_3d.k == 0] = 0
        expected_ky = ifftshift(expected_ky)
        
        # Manual calculation for kz
        expected_kz = mock_kgrid_3d.kz / mock_kgrid_3d.k
        expected_kz[mock_kgrid_3d.k == 0] = 0
        expected_kz = ifftshift(expected_kz)
        
        np.testing.assert_array_almost_equal(result.kx_norm, expected_kx)
        np.testing.assert_array_almost_equal(result.ky_norm, expected_ky)
        np.testing.assert_array_almost_equal(result.kz_norm, expected_kz)
    

    def test_3d_handles_zero_wavenumber(self, mock_record):
        """Test that division by zero is handled correctly in 3D."""
        kgrid = Mock()
        kgrid.dim = 3
        kgrid.kx = np.ones((2, 2, 2))
        kgrid.ky = np.ones((2, 2, 2))
        kgrid.kz = np.ones((2, 2, 2))
        kgrid.k = np.ones((2, 2, 2))
        kgrid.k[0, 0, 0] = 0  # Set one element to zero
        
        result = create_normalized_wavenumber_vectors(
            mock_record, 
            kgrid, 
            is_record_u_split_field=True
        )
        
        assert not np.isnan(result.kx_norm).any()
        assert not np.isnan(result.ky_norm).any()
        assert not np.isnan(result.kz_norm).any()


    def test_ifftshift_applied(self, mock_record):
        """Test that ifftshift is correctly applied."""
        kgrid = Mock()
        kgrid.dim = 1
        kgrid.kx = np.array([1, 2, 3, 4, 5])
        kgrid.k = np.array([1, 1, 1, 1, 1])
        
        result = create_normalized_wavenumber_vectors(
            mock_record, 
            kgrid, 
            is_record_u_split_field=True
        )
        
        # Before ifftshift: [1, 2, 3, 4, 5]
        # After ifftshift (odd length): [3, 4, 5, 1, 2]
        expected = ifftshift(kgrid.kx / kgrid.k)
        np.testing.assert_array_equal(result.kx_norm, expected)
    

    @pytest.mark.parametrize("dim", [1, 2, 3])
    def test_all_dimensions(self, mock_record, dim):
        """Parametrized test for all dimensions."""
        kgrid = Mock()
        kgrid.dim = dim
        
        shape = (5,) * dim
        kgrid.kx = np.random.rand(*shape)
        kgrid.k = np.random.rand(*shape) + 0.1
        
        if dim >= 2:
            kgrid.ky = np.random.rand(*shape)
        if dim == 3:
            kgrid.kz = np.random.rand(*shape)
        
        result = create_normalized_wavenumber_vectors(
            mock_record, 
            kgrid, 
            is_record_u_split_field=True
        )
        
        assert hasattr(result, 'kx_norm')
        if dim >= 2:
            assert hasattr(result, 'ky_norm')
        if dim == 3:
            assert hasattr(result, 'kz_norm')

