#!/usr/bin/env python3
"""
Preprocessing module for RoadTracer project
Handles LISS-IV multispectral imagery preparation and road data clipping
"""

import os
import argparse
import numpy as np
from pathlib import Path
import rasterio
from rasterio.merge import merge
from rasterio.mask import mask
from rasterio.warp import calculate_default_transform, reproject, Resampling
import geopandas as gpd
from shapely.geometry import Point, box
from shapely.ops import unary_union
import warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm
from datetime import datetime
import json

class LISSIVProcessor:
    """Process LISS-IV satellite imagery data"""
    
    def __init__(self, aoi_center, buffer_m=30):
        """
        Initialize processor with Area of Interest
        
        Args:
            aoi_center: tuple (lat, lon) center coordinates
            buffer_m: buffer in meters around roads
        """
        self.aoi_center = aoi_center
        self.buffer_m = buffer_m
        self.crs = 'EPSG:4326'  # WGS84
        
    def create_composite(self, scene_dir, output_path):
        """
        Create RGB composite from LISS-IV bands
        LISS-IV bands: Band2 (Green), Band3 (Red), Band4 (NIR)
        RGB Composite: R=Band3, G=Band2, B=Band2 (false color)
        """
        scene_dir = Path(scene_dir)
        
        # Find band files
        band2_path = scene_dir / 'BAND2.tif'
        band3_path = scene_dir / 'BAND3.tif'
        band4_path = scene_dir / 'BAND4.tif'
        
        if not all([band2_path.exists(), band3_path.exists(), band4_path.exists()]):
            print(f"Missing band files in {scene_dir}")
            return None
            
        # Read bands
        with rasterio.open(band2_path) as b2, \
             rasterio.open(band3_path) as b3, \
             rasterio.open(band4_path) as b4:
            
            # Get metadata from first band
            meta = b2.meta.copy()
            
            # Read band data
            green = b2.read(1)
            red = b3.read(1)
            nir = b4.read(1)
            
            # Create RGB composite (False Color: NIR-R-G)
            rgb = np.stack([nir, red, green], axis=0)
            
            # Update metadata for 3-band output
            meta.update({
                'count': 3,
                'dtype': 'uint16',
                'driver': 'GTiff',
                'compress': 'lzw'
            })
            
            # Write composite
            with rasterio.open(output_path, 'w', **meta) as dst:
                dst.write(rgb)
                
        return output_path
    
    def clip_to_roads(self, image_path, roads_gdf, output_path, buffer_m=None):
        """
        Clip image to road buffer area
        
        Args:
            image_path: path to input image
            roads_gdf: GeoDataFrame with road geometries
            output_path: path for output clipped image
            buffer_m: buffer in meters (overrides class default)
        """
        if buffer_m is None:
            buffer_m = self.buffer_m
            
        # Create buffer around roads
        # Convert to projected CRS for accurate buffering
        roads_proj = roads_gdf.to_crs('EPSG:32643')  # UTM Zone 43N for India
        roads_buffered = roads_proj.buffer(buffer_m)
        roads_union = unary_union(roads_buffered)
        
        # Convert back to image CRS
        with rasterio.open(image_path) as src:
            # Create GeoDataFrame with buffered geometry
            buffer_gdf = gpd.GeoDataFrame(
                [1], 
                geometry=[roads_union], 
                crs='EPSG:32643'
            )
            buffer_gdf = buffer_gdf.to_crs(src.crs)
            
            # Clip image
            out_image, out_transform = mask(
                src, 
                buffer_gdf.geometry, 
                crop=True,
                filled=True,
                nodata=0
            )
            
            # Update metadata
            out_meta = src.meta.copy()
            out_meta.update({
                "driver": "GTiff",
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": out_transform,
                "compress": "lzw"
            })
            
            # Write clipped image
            with rasterio.open(output_path, "w", **out_meta) as dest:
                dest.write(out_image)
                
        return output_path

def preprocess_lissiv_data(input_dir, output_dir, aoi_center, buffer_m, clip_roads):
    """
    Main preprocessing pipeline
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize processor
    processor = LISSIVProcessor(aoi_center, buffer_m)
    
    # Find all scene directories
    scene_dirs = [d for d in (input_dir / 'LISS-IV_images').iterdir() if d.is_dir()]
    
    if not scene_dirs:
        print("No scene directories found!")
        return
        
    print(f"Found {len(scene_dirs)} LISS-IV scenes")
    
    # Process road data if clipping is requested
    roads_gdf = None
    if clip_roads:
        roads_path = input_dir / 'OSM_shapefiles' / 'nh_roads.shp'
        if not roads_path.exists():
            print(f"Road shapefile not found at {roads_path}")
            print("Download NH roads from OpenStreetMap or other sources")
            return
            
        # Load and filter roads
        print("Loading road data...")
        roads_gdf = gpd.read_file(roads_path)
        
        # Filter to National Highways if needed
        if 'highway' in roads_gdf.columns:
            roads_gdf = roads_gdf[roads_gdf['highway'].isin([
                'trunk', 'primary', 'motorway', 'trunk_link', 
                'primary_link', 'motorway_link'
            ])]
        
        # Create AOI bounds (optional - for filtering roads to region)
        if aoi_center:
            # Create a large bounding box around AOI center
            lat, lon = aoi_center
            # Approximately 50km box around center
            aoi_box = box(lon - 0.5, lat - 0.5, lon + 0.5, lat + 0.5)
            aoi_gdf = gpd.GeoDataFrame([1], geometry=[aoi_box], crs='EPSG:4326')
            
            # Clip roads to AOI
            roads_gdf = gpd.clip(roads_gdf, aoi_gdf)
            
        print(f"Filtered to {len(roads_gdf)} road segments")
        
        # Save processed roads
        roads_output = output_dir / 'nh_roads_aoi.shp'
        roads_gdf.to_file(roads_output)
        print(f"Saved filtered roads to {roads_output}")
    
    # Process each scene
    processed_images = []
    metadata = {
        'scenes': [],
        'aoi_center': aoi_center,
        'buffer_m': buffer_m,
        'timestamp': datetime.now().isoformat()
    }
    
    for scene_dir in tqdm(scene_dirs, desc="Processing scenes"):
        scene_name = scene_dir.name
        
        # Create composite
        composite_path = output_dir / f"{scene_name}_composite.tif"
        result = processor.create_composite(scene_dir, composite_path)
        
        if result is None:
            continue
            
        # Clip to roads if requested
        if clip_roads and roads_gdf is not None:
            clipped_path = output_dir / f"{scene_name}_clipped.tif"
            try:
                processor.clip_to_roads(composite_path, roads_gdf, clipped_path)
                processed_images.append(clipped_path)
                
                # Remove unclipped composite to save space
                os.remove(composite_path)
                
                metadata['scenes'].append({
                    'name': scene_name,
                    'output': str(clipped_path),
                    'clipped': True
                })
            except Exception as e:
                print(f"Error clipping {scene_name}: {e}")
                processed_images.append(composite_path)
                metadata['scenes'].append({
                    'name': scene_name,
                    'output': str(composite_path),
                    'clipped': False
                })
        else:
            processed_images.append(composite_path)
            metadata['scenes'].append({
                'name': scene_name,
                'output': str(composite_path),
                'clipped': False
            })
    
    # Save metadata
    metadata_path = output_dir / 'preprocessing_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nPreprocessing complete!")
    print(f"Processed {len(processed_images)} images")
    print(f"Output directory: {output_dir}")
    print(f"Metadata saved to: {metadata_path}")
    
    return processed_images

def main():
    parser = argparse.ArgumentParser(
        description='Preprocess LISS-IV imagery for RoadTracer project'
    )
    
    parser.add_argument(
        '--input_dir',
        type=str,
        required=True,
        help='Input directory containing raw data'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='data/processed',
        help='Output directory for processed data'
    )
    
    parser.add_argument(
        '--aoi_center',
        type=float,
        nargs=2,
        metavar=('LAT', 'LON'),
        help='AOI center coordinates (lat lon)'
    )
    
    parser.add_argument(
        '--buffer',
        type=int,
        default=30,
        help='Buffer distance in meters around roads'
    )
    
    parser.add_argument(
        '--clip',
        action='store_true',
        help='Clip images to road buffer area'
    )
    
    args = parser.parse_args()
    
    # Run preprocessing
    preprocess_lissiv_data(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        aoi_center=tuple(args.aoi_center) if args.aoi_center else None,
        buffer_m=args.buffer,
        clip_roads=args.clip
    )

if __name__ == '__main__':
    main()