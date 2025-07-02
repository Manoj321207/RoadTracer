#!/usr/bin/env python3
"""
Tiler module for RoadTracer project
Tiles large LISS-IV satellite images and corresponding masks into smaller patches
with configurable overlap for training and inference.
"""

import os
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
import rasterio
from rasterio.windows import Window
import json
from datetime import datetime
import shutil

class ImageTiler:
    """Handles tiling of satellite imagery and masks"""
    
    def __init__(self, tile_size=512, overlap=64):
        """
        Initialize tiler
        
        Args:
            tile_size: Size of each tile (square)
            overlap: Overlap between adjacent tiles
        """
        self.tile_size = tile_size
        self.overlap = overlap
        self.stride = tile_size - overlap
        
        if self.stride <= 0:
            raise ValueError("Overlap must be less than tile_size")
        
    def get_tile_coordinates(self, width, height):
        """
        Calculate tile coordinates for given image dimensions
        
        Returns:
            List of tuples (x, y, row, col) for each tile
        """
        tiles = []
        
        # Calculate number of tiles needed
        n_tiles_x = max(1, (width - self.overlap) // self.stride)
        n_tiles_y = max(1, (height - self.overlap) // self.stride)
        
        # Add extra tiles if image is not perfectly divisible
        if (width - self.overlap) % self.stride > 0:
            n_tiles_x += 1
        if (height - self.overlap) % self.stride > 0:
            n_tiles_y += 1
        
        for row in range(n_tiles_y):
            for col in range(n_tiles_x):
                x = col * self.stride
                y = row * self.stride
                
                # Adjust last tiles if they go beyond image boundary
                if x + self.tile_size > width:
                    x = width - self.tile_size
                if y + self.tile_size > height:
                    y = height - self.tile_size
                    
                # Ensure coordinates are valid
                x = max(0, x)
                y = max(0, y)
                
                tiles.append((x, y, row, col))
                
        return tiles
    
    def tile_image(self, image_path, output_dir, is_mask=False, min_valid_ratio=0.1):
        """
        Tile a single image and save tiles
        
        Args:
            image_path: Path to input image
            output_dir: Output directory for tiles
            is_mask: Whether this is a mask (affects filtering)
            min_valid_ratio: Minimum ratio of valid pixels to save tile
            
        Returns:
            List of tile information dictionaries
        """
        
        image_path = Path(image_path)
        base_name = image_path.stem
        
        # Create output directory
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Open the image
        with rasterio.open(image_path) as src:
            # Get image metadata
            meta = src.meta.copy()
            width = src.width
            height = src.height
            
            # Get tile coordinates
            tiles = self.get_tile_coordinates(width, height)
            
            # Update metadata for tiles
            meta.update({
                'width': self.tile_size,
                'height': self.tile_size,
                'driver': 'GTiff',
                'compress': 'lzw'
            })
            
            # Process each tile
            tile_info = []
            saved_count = 0
            
            for x, y, row, col in tqdm(tiles, desc=f"Tiling {base_name}", leave=False):
                # Define window for reading
                actual_width = min(self.tile_size, width - x)
                actual_height = min(self.tile_size, height - y)
                window = Window(x, y, actual_width, actual_height)
                
                # Read the tile data
                tile_data = src.read(window=window)
                
                # Pad if necessary (for edge tiles)
                if actual_width < self.tile_size or actual_height < self.tile_size:
                    padded_data = np.zeros((src.count, self.tile_size, self.tile_size), 
                                          dtype=tile_data.dtype)
                    padded_data[:, :actual_height, :actual_width] = tile_data
                    tile_data = padded_data
                
                # Check if tile has valid data
                if is_mask:
                    # For masks, check if there are any road pixels
                    valid_pixels = np.sum(tile_data > 0)
                    total_pixels = tile_data.size
                    valid_ratio = valid_pixels / total_pixels
                    
                    # Skip empty mask tiles
                    if valid_ratio == 0:
                        continue
                else:
                    # For images, check if not all black/nodata
                    valid_pixels = np.sum(tile_data != 0)
                    total_pixels = tile_data.size
                    valid_ratio = valid_pixels / total_pixels
                    
                    # Skip tiles with too few valid pixels
                    if valid_ratio < min_valid_ratio:
                        continue
                
                # Generate tile filename
                tile_name = f"{base_name}tile{row:03d}_{col:03d}.tif"
                tile_path = output_dir / tile_name
                
                # Update transform for the tile
                transform = src.window_transform(window)
                meta.update({'transform': transform})
                
                # Write tile
                with rasterio.open(tile_path, 'w', **meta) as dst:
                    dst.write(tile_data)
                
                saved_count += 1
                
                # Store tile information
                tile_info.append({
                    'filename': tile_name,
                    'parent_image': image_path.name,
                    'row': row,
                    'col': col,
                    'x': x,
                    'y': y,
                    'actual_width': actual_width,
                    'actual_height': actual_height,
                    'tile_size': self.tile_size,
                    'valid_ratio': float(valid_ratio)
                })
        
        return tile_info, saved_count

def tile_dataset(input_dir, mask_dir, output_dir, tile_size, overlap, 
                min_valid_ratio=0.1, skip_empty_masks=True):
    """
    Tile all images and corresponding masks in the dataset
    
    Args:
        input_dir: Directory containing processed images
        mask_dir: Directory containing masks
        output_dir: Output directory for tiles
        tile_size: Size of each tile
        overlap: Overlap between tiles
        min_valid_ratio: Minimum ratio of valid pixels
        skip_empty_masks: Whether to skip tiles with empty masks
    """
    
    input_dir = Path(input_dir)
    mask_dir = Path(mask_dir) if mask_dir else None
    output_dir = Path(output_dir)
    
    # Create output directories
    image_output_dir = output_dir / 'images'
    mask_output_dir = output_dir / 'masks'
    
    image_output_dir.mkdir(parents=True, exist_ok=True)
    if mask_dir:
        mask_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize tiler
    tiler = ImageTiler(tile_size=tile_size, overlap=overlap)
    
    # Find all image files
    image_extensions = ['.tif', '.tiff', '.TIF', '.TIFF']
    image_files = []
    for ext in image_extensions:
        image_files.extend(input_dir.glob(f'*{ext}'))
    
    # Sort for consistent ordering
    image_files = sorted(image_files)
    
    if not image_files:
        print(f"No image files found in {input_dir}")
        return
    
    print(f"Found {len(image_files)} images to tile")
    print(f"Tile size: {tile_size}x{tile_size}")
    print(f"Overlap: {overlap} pixels")
    print(f"Stride: {tile_size - overlap} pixels")
    
    # Process each image
    all_tiles_info = {
        'images': [],
        'masks': []
    }
    
    total_image_tiles = 0
    total_mask_tiles = 0
    paired_tiles = 0
    
    for image_path in tqdm(image_files, desc="Processing images"):
        # Tile the image
        image_tiles_info, image_count = tiler.tile_image(
            image_path, image_output_dir, is_mask=False, 
            min_valid_ratio=min_valid_ratio
        )
        all_tiles_info['images'].extend(image_tiles_info)
        total_image_tiles += image_count
        
        # Check for corresponding mask
        if mask_dir:
            # Try different mask naming conventions
            mask_names = [
                image_path.stem + '_mask.tif',
                image_path.stem + '.tif',
                image_path.name
            ]
            
            mask_path = None
            for mask_name in mask_names:
                potential_mask = mask_dir / mask_name
                if potential_mask.exists():
                    mask_path = potential_mask
                    break
            
            if mask_path:
                # Tile the mask with same coordinates
                mask_tiles_info, mask_count = tiler.tile_image(
                    mask_path, mask_output_dir, is_mask=True,
                    min_valid_ratio=0.0  # Don't filter masks by valid ratio
                )
                all_tiles_info['masks'].extend(mask_tiles_info)
                total_mask_tiles += mask_count
                
                # If skip_empty_masks, remove image tiles without corresponding masks
                if skip_empty_masks:
                    # Get mask tile names
                    mask_tile_names = {info['filename'] for info in mask_tiles_info}
                    
                    # Remove unpaired image tiles
                    for img_tile_info in image_tiles_info:
                        img_tile_name = img_tile_info['filename']
                        expected_mask_name = img_tile_name.replace(image_path.stem, mask_path.stem)
                        
                        if expected_mask_name not in mask_tile_names:
                            # Remove the unpaired image tile
                            img_tile_path = image_output_dir / img_tile_name
                            if img_tile_path.exists():
                                os.remove(img_tile_path)
                                total_image_tiles -= 1
                        else:
                            paired_tiles += 1
                
                print(f"  - Tiled mask: {mask_path.name} ({mask_count} tiles)")
            else:
                print(f"  - No mask found for {image_path.name}")
    
    # Create train/val split
    if paired_tiles > 0:
        create_train_val_split(all_tiles_info, output_dir, split_ratio=0.8)
    
    # Save tiling metadata
    metadata = {
        'tile_size': tile_size,
        'overlap': overlap,
        'stride': tile_size - overlap,
        'total_image_tiles': total_image_tiles,
        'total_mask_tiles': total_mask_tiles,
        'paired_tiles': paired_tiles,
        'source_images': len(image_files),
        'min_valid_ratio': min_valid_ratio,
        'skip_empty_masks': skip_empty_masks,
        'timestamp': datetime.now().isoformat(),
        'tiles_info': all_tiles_info
    }
    
    metadata_path = output_dir / 'tiling_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nTiling complete!")
    print(f"Total image tiles created: {total_image_tiles}")
    print(f"Total mask tiles created: {total_mask_tiles}")
    print(f"Paired tiles (image+mask): {paired_tiles}")
    print(f"Metadata saved to: {metadata_path}")
    
    # Print summary statistics
    print(f"\nSummary:")
    print(f"  - Output directory: {output_dir}")
    print(f"  - Images directory: {image_output_dir}")
    print(f"  - Masks directory: {mask_output_dir}")

def create_train_val_split(tiles_info, output_dir, split_ratio=0.8):
    """
    Create train/validation split file
    
    Args:
        tiles_info: Dictionary containing tile information
        output_dir: Output directory
        split_ratio: Ratio of training data (default: 0.8)
    """
    import random
    
    output_dir = Path(output_dir)
    
    # Get paired tiles (both image and mask exist)
    image_tiles = {info['filename']: info for info in tiles_info.get('images', [])}
    mask_tiles = {info['filename']: info for info in tiles_info.get('masks', [])}
    
    # Find paired tiles
    paired_tiles = []
    for img_filename, img_info in image_tiles.items():
        # Check if corresponding mask exists
        mask_filename = img_filename.replace('_composite', '_composite_mask')
        mask_filename = mask_filename.replace('_clipped', '_clipped_mask')
        
        if mask_filename in mask_tiles:
            paired_tiles.append({
                'image': img_filename,
                'mask': mask_filename,
                'parent': img_info['parent_image']
            })
    
    if not paired_tiles:
        print("No paired tiles found for train/val split")
        return
    
    # Group tiles by parent image to avoid data leakage
    tiles_by_parent = {}
    for tile in paired_tiles:
        parent = tile['parent']
        if parent not in tiles_by_parent:
            tiles_by_parent[parent] = []
        tiles_by_parent[parent].append(tile)
    
    # Split by parent images
    parent_images = list(tiles_by_parent.keys())
    random.shuffle(parent_images)
    
    n_train_parents = int(len(parent_images) * split_ratio)
    train_parents = parent_images[:n_train_parents]
    val_parents = parent_images[n_train_parents:]
    
    # Collect tiles for each split
    train_tiles = []
    val_tiles = []
    
    for parent in train_parents:
        train_tiles.extend(tiles_by_parent[parent])
    
    for parent in val_parents:
        val_tiles.extend(tiles_by_parent[parent])
    
    # Save split information
    split_info = {
        'train': train_tiles,
        'validation': val_tiles,
        'train_count': len(train_tiles),
        'val_count': len(val_tiles),
        'train_parents': train_parents,
        'val_parents': val_parents,
        'split_ratio': split_ratio,
        'timestamp': datetime.now().isoformat()
    }
    
    split_path = output_dir / 'train_val_split.json'
    with open(split_path, 'w') as f:
        json.dump(split_info, f, indent=2)
    
    print(f"\nTrain/Val split created:")
    print(f"  - Training tiles: {len(train_tiles)} from {len(train_parents)} images")
    print(f"  - Validation tiles: {len(val_tiles)} from {len(val_parents)} images")
    print(f"  - Split info saved to: {split_path}")

def verify_tile_pairs(tiles_dir):
    """
    Verify that image and mask tiles are properly paired
    
    Args:
        tiles_dir: Directory containing tiled data
    """
    tiles_dir = Path(tiles_dir)
    images_dir = tiles_dir / 'images'
    masks_dir = tiles_dir / 'masks'
    
    if not images_dir.exists() or not masks_dir.exists():
        print("Images or masks directory not found")
        return
    
    # Get all tiles
    image_tiles = set(f.name for f in images_dir.glob('*.tif'))
    mask_tiles = set(f.name for f in masks_dir.glob('*.tif'))
    
    # Check pairing
    paired = 0
    unpaired_images = []
    unpaired_masks = []
    
    for img_tile in image_tiles:
        # Expected mask name
        mask_tile = img_tile.replace('_composite', '_composite_mask')
        mask_tile = mask_tile.replace('_clipped', '_clipped_mask')
        
        if mask_tile in mask_tiles:
            paired += 1
        else:
            unpaired_images.append(img_tile)
    
    for mask_tile in mask_tiles:
        # Expected image name
        img_tile = mask_tile.replace('_mask', '')
        
        if img_tile not in image_tiles:
            unpaired_masks.append(mask_tile)
    
    print(f"\nTile Verification Results:")
    print(f"  - Total image tiles: {len(image_tiles)}")
    print(f"  - Total mask tiles: {len(mask_tiles)}")
    print(f"  - Paired tiles: {paired}")
    print(f"  - Unpaired images: {len(unpaired_images)}")
    print(f"  - Unpaired masks: {len(unpaired_masks)}")
    
    if unpaired_images:
        print(f"\nFirst 5 unpaired images: {unpaired_images[:5]}")
    if unpaired_masks:
        print(f"First 5 unpaired masks: {unpaired_masks[:5]}")

def visualize_tiles(tiles_dir, num_samples=5):
    """
    Visualize sample tiles for quality check
    """
    import matplotlib.pyplot as plt
    
    tiles_dir = Path(tiles_dir)
    images_dir = tiles_dir / 'images'
    masks_dir = tiles_dir / 'masks'
    vis_dir = tiles_dir / 'visualizations'
    vis_dir.mkdir(exist_ok=True)
    
    # Load train/val split if available
    split_path = tiles_dir / 'train_val_split.json'
    if split_path.exists():
        with open(split_path) as f:
            split_info = json.load(f)
        tile_pairs = split_info['train'][:num_samples]
    else:
        # Just get first few tiles
        image_tiles = sorted(images_dir.glob('*.tif'))[:num_samples]
        tile_pairs = []
        for img_path in image_tiles:
            mask_name = img_path.name.replace('_composite', '_composite_mask')
            mask_name = mask_name.replace('_clipped', '_clipped_mask')
            tile_pairs.append({
                'image': img_path.name,
                'mask': mask_name
            })
    
    print(f"\nCreating tile visualizations...")
    
    for i, pair in enumerate(tile_pairs):
        img_path = images_dir / pair['image']
        mask_path = masks_dir / pair['mask']
        
        if not img_path.exists() or not mask_path.exists():
            continue
        
        # Read tiles
        with rasterio.open(img_path) as src:
            if src.count >= 3:
                img_data = np.stack([src.read(i) for i in range(1, 4)], axis=-1)
            else:
                band = src.read(1)
                img_data = np.stack([band, band, band], axis=-1)
            
            # Normalize
            img_min, img_max = img_data.min(), img_data.max()
            if img_max > img_min:
                img_data = (img_data - img_min) / (img_max - img_min)
        
        with rasterio.open(mask_path) as src:
            mask_data = src.read(1)
        
        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        
        # Image
        axes[0].imshow(img_data)
        axes[0].set_title(f'Image Tile {i+1}')
        axes[0].axis('off')
        
        # Mask
        axes[1].imshow(mask_data, cmap='binary')
        axes[1].set_title('Mask')
        axes[1].axis('off')
        
        # Overlay
        overlay = img_data.copy()
        mask_overlay = np.zeros_like(img_data)
        mask_overlay[:, :, 0] = mask_data  # Red for roads
        overlay = overlay * 0.7 + mask_overlay * 0.3
        axes[2].imshow(overlay)
        axes[2].set_title('Overlay')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(vis_dir / f'tile_sample_{i+1}.png', dpi=150)
        plt.close()
    
    print(f"Visualizations saved to: {vis_dir}")

def main():
    parser = argparse.ArgumentParser(
        description='Tile LISS-IV satellite images and masks for RoadTracer project'
    )
    
    parser.add_argument(
        '--input_dir',
        type=str,
        required=True,
        help='Directory containing processed images'
    )
    
    parser.add_argument(
        '--mask_dir',
        type=str,
        default=None,
        help='Directory containing masks (if not provided, only images will be tiled)'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Output directory for tiles'
    )
    
    parser.add_argument(
        '--tile_size',
        type=int,
        default=512,
        help='Size of each tile (default: 512)'
    )
    
    parser.add_argument(
        '--overlap',
        type=int,
        default=64,
        help='Overlap between tiles in pixels (default: 64)'
    )
    
    parser.add_argument(
        '--min_valid_ratio',
        type=float,
        default=0.1,
        help='Minimum ratio of valid pixels to save tile (default: 0.1)'
    )
    
    parser.add_argument(
        '--keep_empty_masks',
        action='store_true',
        help='Keep tiles even if corresponding mask is empty'
    )
    
    parser.add_argument(
        '--verify',
        action='store_true',
        help='Verify tile pairs after tiling'
    )
    
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Create visualization samples'
    )
    
    parser.add_argument(
        '--vis_samples',
        type=int,
        default=5,
        help='Number of visualization samples (default: 5)'
    )
    
    args = parser.parse_args()
    
    # Set mask_dir to default if not provided
    if args.mask_dir is None:
        # Assume masks are in ../masks relative to input_dir
        input_path = Path(args.input_dir)
        potential_mask_dir = input_path.parent / 'masks'
        if potential_mask_dir.exists():
            args.mask_dir = str(potential_mask_dir)
            print(f"Using mask directory: {args.mask_dir}")
    
    print("Starting tiling process...")
    print(f"Input directory: {args.input_dir}")
    print(f"Mask directory: {args.mask_dir if args.mask_dir else 'None'}")
    print(f"Output directory: {args.output_dir}")
    print(f"Tile size: {args.tile_size}x{args.tile_size}")
    print(f"Overlap: {args.overlap} pixels")
    print(f"Min valid ratio: {args.min_valid_ratio}")
    print("-" * 50)
    
    # Run tiling
    tile_dataset(
        input_dir=args.input_dir,
        mask_dir=args.mask_dir,
        output_dir=args.output_dir,
        tile_size=args.tile_size,
        overlap=args.overlap,
        min_valid_ratio=args.min_valid_ratio,
        skip_empty_masks=not args.keep_empty_masks
    )
    
    # Verify if requested
    if args.verify:
        print("\nVerifying tile pairs...")
        verify_tile_pairs(args.output_dir)
    
    # Visualize if requested
    if args.visualize:
        visualize_tiles(args.output_dir, num_samples=args.vis_samples)

if __name__ == '__main__':
    main()
    