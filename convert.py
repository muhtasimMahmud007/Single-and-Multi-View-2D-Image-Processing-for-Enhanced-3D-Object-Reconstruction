import os
import numpy as np
import trimesh
import torch
from pathlib import Path

def save_binvox(voxel_data, filepath, scale=1.0, translate=None):
    dims = voxel_data.shape
    translate = translate if translate is not None else [0, 0, 0]
    with open(filepath, 'wb') as f:
        f.write(b'#binvox 1\n')
        f.write(f'dim {dims[0]} {dims[1]} {dims[2]}\n'.encode('ascii'))
        f.write(f'translate {translate[0]} {translate[1]} {translate[2]}\n'.encode('ascii'))
        f.write(f'scale {scale}\n'.encode('ascii'))
        f.write(b'data\n')
        voxels_flat = voxel_data.flatten()
        start = 0
        while start < len(voxels_flat):
            val = voxels_flat[start]
            count = 0
            while start + count < len(voxels_flat) and voxels_flat[start + count] == val and count < 255:
                count += 1
            f.write(bytes([val, count]))
            start += count

def voxelize_mesh(mesh, resolution=32):
    points = mesh.sample(resolution ** 3)
    grid = torch.zeros((resolution, resolution, resolution), dtype=torch.uint8, device='cpu')
    points = torch.tensor(points, device='cpu')
    indices = ((points + 1) * (resolution / 2)).long()
    indices = indices.clamp(0, resolution - 1)  # Ensure indices are within bounds
    grid[indices[:, 0], indices[:, 1], indices[:, 2]] = 1
    return grid.cpu().numpy()

def main():
    directory = r"D:\plusplus\2002"
    log_file = r"D:\plusplus\log.txt"
    resolution = 128

    subdirs = [x for x in Path(directory).iterdir() if x.is_dir()]

    count = 0
    for subdir in subdirs:
        pathlist = Path(subdir).glob("**/*.obj")
        for path in pathlist:
            try:
                # Load the mesh using trimesh
                mesh = trimesh.load(str(path))
                # Voxelize the mesh using GPU
                voxel_grid = voxelize_mesh(mesh, resolution=resolution)
                # Save the voxel grid to a .binvox file
                binvox_file = path.with_suffix('.binvox')
                save_binvox(voxel_grid, str(binvox_file))
                print(f".binvox file created: {binvox_file}")
                with open(log_file, "a") as f:
                    f.write(f"Processed {path} successfully. .binvox file created at {binvox_file}\n")
                    f.write("=" * 30 + "\n")
            except Exception as e:
                print(f"An error occurred while converting {path}: {e}")
                with open(log_file, "a") as f:
                    f.write(f"An error occurred while converting {path}: {e}\n")
                    f.write("=" * 30 + "\n")
        
        count += 1
        print(f"{count}/{len(subdirs)}")

    print("DONE")

if __name__ == "__main__":
    main()
