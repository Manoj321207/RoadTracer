#!/usr/bin/env python3
"""
Data Generation module for RoadTracer project
Generates segmentation masks from OpenStreetMap road shapefiles
aligned with LISS-IV satellite imagery
"""

import os
import argparse
import numpy as np
from pathlib import Path
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_bounds
import geopandas as gpd
from shapely.geometry import LineString, MultiLineString, box
from shapely.ops import unary_union
import cv2
from tqdm import tqdm
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class MaskGenerator:
    """Generate road segmentation masks from vector data"""
    
    def __init__(self, road_width=5, smooth=True):
        """
        Initialize mask generator
        
        Args:
            road_width: Width of roads in pixels (default: 5)
            smooth: Whether to apply smoothing to masks
        """
        self.road_width = road_width
        self.smooth = smooth
        
    def generate_mask_from_shapefile(self, image_path, roads_gdf, output_path):
        """
        Generate binary mask from road shapefile aligned with image
        
        Args:
            image_path: Path to reference image
            roads_gdf: GeoDataFrame containing road geometries
            output_path: Path to save mask
        """
        # Open reference image to get metadata
        with rasterio.open(image_path) as src:
            # Get image properties
            height = src.height
            width = src.width
            transform = src.transform
            crs = src.crs
            
            # Reproject roads to image CRS if needed
            if roads_gdf.crs != crs:
                roads_gdf = roads_gdf.to_crs(crs)
            
            # Filter roads that intersect with image bounds
            image_bounds = src.bounds
            image_bbox = box(image_bounds.left, image_bounds.bottom, 
                           image_bounds.right, image_bounds.top)
            
            # Clip roads to image extent
            roads_clipped = roads_gdf[roads_gdf.intersects(image_bbox)]
            
            if len(roads_clipped) == 0:
                print(f"No roads found in image extent: {Path(image_path).name}")
                # Create empty mask
                mask = np.zeros((height, width), dtype=np.uint8)
            else:
                # Buffer road lines to create width
                # Convert to metric CRS for accurate buffering
                roads_metric = roads_clipped.to_crs('EPSG:32643')  # UTM Zone 43N
                
                # Calculate buffer based on image resolution
                # LISS-IV resolution is 5.8m, so for 5-pixel width: ~15m buffer
                pixel_size = 5.8  # meters
                buffer_distance = (self.road_width * pixel_size) / 2
                
                # Buffer geometries
                roads_buffered = roads_metric.geometry.buffer(buffer_distance)
                
                # Convert back to image CRS
                roads_buffered_gdf = gpd.GeoDataFrame(
                    geometry=roads_buffered, 
                    crs='EPSG:32643'
                )
                roads_buffered_gdf = roads_buffered_gdf.to_crs(crs)
                
                # Rasterize buffered roads
                mask = rasterize(
                    [(geom, 1) for geom in roads_buffered_gdf.geometry],
                    out_shape=(height, width),
                    transform=transform,
                    fill=0,
                    dtype=np.uint8
                )
            
            # Apply morphological operations for smoothing
            if self.smooth and np.any(mask):
                # Close small gaps
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
                
                # Smooth edges
                mask = cv2.GaussianBlur(mask, (3, 3), 0)
                mask = (mask > 0.5).astype(np.uint8)
            
            # Save mask
            profile = src.profile.copy()
            profile.update({
                'count': 1,
                'dtype': 'uint8',
                'compress': 'lzw',
                'nodata': None
            })
            
            with rasterio.open(output_path, 'w', **profile) as dst:
                dst.write(mask, 1)
                
        return output_path
    
    def generate_centerline_mask(self, image_path, roads_gdf, output_path):
        """
        Generate thin centerline mask (1-2 pixels wide)
        
        Args:
            image_path: Path to reference image
            roads_gdf: GeoDataFrame containing road geometries
            output_path: Path to save mask
        """
        # Open reference image
        with rasterio.open(image_path) as src:
            height = src.height
            width = src.width
            transform = src.transform
            crs = src.crs
            
            # Reproject roads if needed
            if roads_gdf.crs != crs:
                roads_gdf = roads_gdf.to_crs(crs)
            
            # Filter roads that intersect with image bounds
            image_bounds = src.bounds
            image_bbox = box(image_bounds.left, image_bounds.bottom, 
                           image_bounds.right, image_bounds.top)
            
            # Clip roads to image extent
            roads_clipped = roads_gdf[roads_gdf.intersects(image_bbox)]
            
            if len(roads_clipped) == 0:
                print(f"No roads found in image extent: {Path(image_path).name}")
                # Create empty mask
                mask = np.zeros((height, width), dtype=np.uint8)
            else:
                # Rasterize centerlines directly (thin lines)
                mask = rasterize(
                    [(geom, 1) for geom in roads_clipped.geometry],
                    out_shape=(height, width),
                    transform=transform,
                    fill=0,
                    dtype=np.uint8,
                    all_touched=True  # Include all pixels touched by line
                )
                
                # Optional: dilate slightly for 2-pixel width
                if self.road_width > 1:
                    kernel = np.ones((2, 2), np.uint8)
                    mask = cv2.dilate(mask, kernel, iterations=1)
            
            # Save mask
            profile = src.profile.copy()
            profile.update({
                'count': 1,
                'dtype': 'uint8',
                'compress': 'lzw'
            })
            
            with rasterio.open(output_path, 'w', **profile) as dst:
                dst.write(mask, 1)
                
        return output_path

def process_dataset(imagery_dir, shp_file, output_dir, road_width=5, 
                   centerline=False, smooth=True):
    """
    Process all images to generate masks
    
    Args:
        imagery_dir: Directory containing processed images
        shp_file: Path to road shapefile
        output_dir: Output directory for masks
        road_width: Width of roads in pixels
        centerline: Generate thin centerline masks instead of buffered
        smooth: Apply smoothing to masks
    """
    imagery_dir = Path(imagery_dir)
    output_dir = Path(output_dir)
    
    # Create output directories
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load road shapefile
    print(f"Loading road data from {shp_file}...")
    try:
        roads_gdf = gpd.read_file(shp_file)
        print(f"Loaded {len(roads_gdf)} road segments")
    except Exception as e:
        print(f"Error loading shapefile: {e}")
        return
    
    # Filter to National Highways if needed
    if 'highway' in roads_gdf.columns:
        # Keep only major roads
        major_roads = ['trunk', 'primary', 'motorway', 'trunk_link', 
                      'primary_link', 'motorway_link']
        roads_gdf = roads_gdf[roads_gdf['highway'].isin(major_roads)]
        print(f"Filtered to {len(roads_gdf)} National Highway segments")
    elif 'ref' in roads_gdf.columns:
        # Alternative: filter by NH reference
        roads_gdf = roads_gdf[roads_gdf['ref'].str.contains('NH', na=False)]
        print(f"Filtered to {len(roads_gdf)} NH road segments")
    
    if len(roads_gdf) == 0:
        print("No valid road segments found after filtering!")
        return
    
    # Initialize mask generator
    generator = MaskGenerator(road_width=road_width, smooth=smooth)
    
    # Find all processed images
    image_patterns = ['_clipped.tif', '_composite.tif', '*.tif']
    image_files = []
    for pattern in image_patterns:
        image_files.extend(list(imagery_dir.glob(pattern)))
    
    # Remove duplicates and sort
    image_files = sorted(list(set(image_files)))
    
    if not image_files:
        print(f"No image files found in {imagery_dir}")
        return
    
    print(f"Found {len(image_files)} images to process")
    
    # Process statistics
    stats = {
        'total_images': len(image_files),
        'successful': 0,
        'failed': 0,
        'empty_masks': 0,
        'road_pixels_stats': []
    }
    
    # Process each image
    for image_path in tqdm(image_files, desc="Generating masks"):
        try:
            # Generate output filename
            mask_name = image_path.stem + '_mask.tif'
            mask_path = output_dir / mask_name
            
            # Generate mask
            if centerline:
                generator.generate_centerline_mask(image_path, roads_gdf, mask_path)
            else:
                generator.generate_mask_from_shapefile(image_path, roads_gdf, mask_path)
            
            # Verify mask
            with rasterio.open(mask_path) as src:
                mask_data = src.read(1)
                road_pixels = np.sum(mask_data > 0)
                total_pixels = mask_data.size
                road_percentage = (road_pixels / total_pixels) * 100
                
                stats['road_pixels_stats'].append({
                    'image': image_path.name,
                    'road_pixels': int(road_pixels),
                    'total_pixels': int(total_pixels),
                    'road_percentage': float(road_percentage)
                })
                
                if road_pixels == 0:
                    stats['empty_masks'] += 1
                    print(f"\nWarning: Empty mask generated for {image_path.name}")
            
            stats['successful'] += 1
            
        except Exception as e:
            print(f"\nError processing {image_path.name}: {e}")
            stats['failed'] += 1
    
    # Calculate statistics
    if stats['road_pixels_stats']:
        road_percentages = [s['road_percentage'] for s in stats['road_pixels_stats']]
        stats['avg_road_percentage'] = float(np.mean(road_percentages))
        stats['min_road_percentage'] = float(np.min(road_percentages))
        stats['max_road_percentage'] = float(np.max(road_percentages))
    
    # Save metadata and statistics
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'imagery_dir': str(imagery_dir),
        'shapefile': str(shp_file),
        'road_width_pixels': road_width,
        'centerline_mode': centerline,
        'smoothing': smooth,
        'statistics': stats
    }
    
    metadata_path = output_dir / 'mask_generation_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Print summary
    print(f"\nMask generation complete!")
    print(f"Successfully processed: {stats['successful']} images")
    print(f"Failed: {stats['failed']} images")
    print(f"Empty masks: {stats['empty_masks']}")
    if 'avg_road_percentage' in stats:
        print(f"Average road coverage: {stats['avg_road_percentage']:.2f}%")
    print(f"Output directory: {output_dir}")
    print(f"Metadata saved to: {metadata_path}")

def visualize_samples(imagery_dir, mask_dir, output_dir, num_samples=5):
    """
    Create visualization of image-mask pairs for verification
    """
    from matplotlib import pyplot as plt
    
    imagery_dir = Path(imagery_dir)
    mask_dir = Path(mask_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find matching pairs
    image_files = sorted(list(imagery_dir.glob('*.tif')))[:num_samples]
    
    if not image_files:
        print("No images found for visualization")
        return
    
    print(f"Creating {len(image_files)} visualization samples...")
    
    for i, img_path in enumerate(image_files):
        mask_name = img_path.stem + '_mask.tif'
        mask_path = mask_dir / mask_name
        
        if not mask_path.exists():
            print(f"Mask not found for {img_path.name}, skipping...")
            continue
            
        # Read image and mask
        with rasterio.open(img_path) as src:
            # Read RGB bands
            if src.count >= 3:
                # Read first 3 bands as RGB
                img_data = np.stack([src.read(i) for i in range(1, 4)], axis=-1)
            else:
                # Single band - replicate to RGB
                band = src.read(1)
                img_data = np.stack([band, band, band], axis=-1)
            
            # Normalize for display
            img_min = img_data.min()
            img_max = img_data.max()
            if img_max > img_min:
                img_data = (img_data - img_min) / (img_max - img_min)
            else:
                img_data = np.zeros_like(img_data)
        
        with rasterio.open(mask_path) as src:
            mask_data = src.read(1)
        
        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        axes[0].imshow(img_data)
        axes[0].set_title(f'Image: {img_path.name}')
        axes[0].axis('off')
        
        # Mask
        axes[1].imshow(mask_data, cmap='binary')
        axes[1].set_title('Generated Mask')
        axes[1].axis('off')
        
        # Overlay
        overlay = img_data.copy()
        mask_colored = np.zeros_like(img_data)
        mask_colored[:, :, 0] = mask_data * 255  # Red channel for roads
        overlay = cv2.addWeighted(overlay, 0.7, mask_colored, 0.3, 0)
        axes[2].imshow(overlay)
        axes[2].set_title('Overlay')
        axes[2].axis('off')
        
        plt.tight_layout()
        output_path = output_dir / f'visualization_{i+1}.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved visualization to {output_path}")

def main():
    parser = argparse.ArgumentParser(
        description='Generate road segmentation masks for RoadTracer project'
    )
    
    parser.add_argument(
        '--imagery_dir',
        type=str,
        required=True,
        help='Directory containing processed LISS-IV images'
    )
    
    parser.add_argument(
        '--shp_file',
        type=str,
        required=True,
        help='Path to road shapefile (e.g., nh_roads.shp)'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Output directory for generated masks'
    )
    
    parser.add_argument(
        '--road_width',
        type=int,
        default=5,
        help='Width of roads in pixels (default: 5)'
    )
    
    parser.add_argument(
        '--centerline',
        action='store_true',
        help='Generate thin centerline masks instead of buffered roads'
    )
    
    parser.add_argument(
        '--no_smooth',
        action='store_true',
        help='Disable smoothing of masks'
    )
    
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Generate visualization samples after mask generation'
    )
    
    parser.add_argument(
        '--vis_samples',
        type=int,
        default=5,
        help='Number of visualization samples to generate (default: 5)'
    )
    
    args = parser.parse_args()
    
    # Process the dataset
    print("Starting mask generation process...")
    print(f"Input imagery: {args.imagery_dir}")
    print(f"Road shapefile: {args.shp_file}")
    print(f"Output directory: {args.output_dir}")
    print(f"Road width: {args.road_width} pixels")
    print(f"Mode: {'Centerline' if args.centerline else 'Buffered'}")
    print(f"Smoothing: {'Disabled' if args.no_smooth else 'Enabled'}")
    print("-" * 50)
    
    process_dataset(
        imagery_dir=args.imagery_dir,
        shp_file=args.shp_file,
        output_dir=args.output_dir,
        road_width=args.road_width,
        centerline=args.centerline,
        smooth=not args.no_smooth
    )
    
    # Generate visualizations if requested
    if args.visualize:
        print("\nGenerating visualization samples...")
        vis_output_dir = Path(args.output_dir).parent / 'output/visualizations'
        visualize_samples(
            imagery_dir=args.imagery_dir,
            mask_dir=args.output_dir,
            output_dir=vis_output_dir,
            num_samples=args.vis_samples
        )
        print(f"Visualizations saved to: {vis_output_dir}")

if __name__ == '__main__':
    main()