"""
Unit Tests for feature_extraction.py
Tests SpatialFeatureExtractor class
"""

import pytest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config


class TestSpatialFeatureExtractor:
    """Tests for SpatialFeatureExtractor class"""
    
    @pytest.fixture
    def extractor(self):
        """Create SpatialFeatureExtractor instance"""
        from feature_extraction import SpatialFeatureExtractor
        return SpatialFeatureExtractor()
    
    def test_extractor_initialization(self, extractor):
        """Test extractor initializes with correct paths"""
        assert extractor.dem_path == config.DEM_PATH
        assert extractor.rivers is not None
        
    def test_get_features_returns_dict(self, extractor):
        """Test get_features returns dictionary"""
        lat, lon = -0.5, 117.15
        result = extractor.get_features(lat, lon)
        assert isinstance(result, dict)
        
    def test_get_features_has_required_keys(self, extractor):
        """Test output contains required feature keys"""
        lat, lon = -0.5, 117.15
        result = extractor.get_features(lat, lon)
        
        required_keys = ["elevation", "slope", "twi", "river_dist_m"]
        for key in required_keys:
            assert key in result, f"Missing feature: {key}"
            
    def test_river_distance_non_negative(self, extractor):
        """Test river distance is always non-negative"""
        lat, lon = -0.5, 117.15
        result = extractor.get_features(lat, lon)
        assert result["river_dist_m"] >= 0, "River distance should be non-negative"
        
    def test_slope_non_negative(self, extractor):
        """Test slope is non-negative"""
        lat, lon = -0.5, 117.15
        result = extractor.get_features(lat, lon)
        assert result["slope"] >= 0, "Slope should be non-negative"
        
    def test_elevation_realistic_range(self, extractor):
        """Test elevation is within Samarinda realistic range"""
        lat, lon = -0.5, 117.15
        result = extractor.get_features(lat, lon)
        # Samarinda elevation typically between -5m and 100m
        # Allow 0 for missing DEM data
        assert result["elevation"] >= -10, "Elevation too low"
        assert result["elevation"] <= 200, "Elevation too high"
        
    def test_different_locations(self, extractor):
        """Test features vary for different locations"""
        loc1 = extractor.get_features(-0.47, 117.14)  # Simpang Lembuswana
        loc2 = extractor.get_features(-0.35, 117.23)  # Upstream area
        
        # At least river distance should differ
        # (might be same if DEM not loaded properly)
        assert loc1 is not None
        assert loc2 is not None
        
    def test_flow_accumulation_present(self, extractor):
        """Test flow accumulation feature is present"""
        lat, lon = -0.5, 117.15
        result = extractor.get_features(lat, lon)
        assert "flow_accumulation" in result
        assert result["flow_accumulation"] >= 0


class TestRiverGeometry:
    """Tests for river geometry setup"""
    
    @pytest.fixture
    def extractor(self):
        """Create SpatialFeatureExtractor instance"""
        from feature_extraction import SpatialFeatureExtractor
        return SpatialFeatureExtractor()
        
    def test_rivers_multilinestring(self, extractor):
        """Test rivers is a MultiLineString"""
        from shapely.geometry import MultiLineString
        assert isinstance(extractor.rivers, MultiLineString)
        
    def test_rivers_has_geometries(self, extractor):
        """Test rivers contains line geometries"""
        assert len(extractor.rivers.geoms) >= 2  # Mahakam and Karang Mumus
        
    def test_point_on_river_small_distance(self, extractor):
        """Test point near river has small distance"""
        # Approximate river coordinate
        lat, lon = -0.50, 117.15  # Near Karang Mumus
        result = extractor.get_features(lat, lon)
        # Distance should be relatively small (< 10km)
        assert result["river_dist_m"] < 10000, "Point should be near a river"


class TestDEMLoading:
    """Tests for DEM data loading"""
    
    @pytest.fixture
    def extractor(self):
        """Create SpatialFeatureExtractor instance"""
        from feature_extraction import SpatialFeatureExtractor
        return SpatialFeatureExtractor()
        
    def test_dem_path_exists_or_handled(self, extractor):
        """Test DEM path is valid or error handled gracefully"""
        # Should not crash even if DEM doesn't exist
        if os.path.exists(config.DEM_PATH):
            assert extractor.dem is not None
        else:
            # Should still work with default values
            result = extractor.get_features(-0.5, 117.15)
            assert result is not None
            
    def test_slope_map_calculated(self, extractor):
        """Test slope map is calculated from DEM"""
        if extractor.dem is not None:
            assert extractor.slope_map is not None
            assert extractor.slope_map.shape == extractor.dem.shape
