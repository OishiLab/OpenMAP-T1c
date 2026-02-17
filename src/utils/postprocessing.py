import os
import pickle
from scipy import ndimage
import numpy as np
import torch

# Get the absolute path of the current file (postprocessing.py)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Construct the absolute path to the 'split_map.pkl' lookup table
SPLIT_MAP_PATH = os.path.join(CURRENT_DIR, "split_map.pkl")

def insert_label(seg, sn, fn):
    new_seg = seg.copy()
    for num in reversed(range(sn, fn)):
        new_seg[seg == num] = num + 6
    return new_seg

def relabel_segmentation(parc):
    seg = parc.copy()
    seg = insert_label(seg, 251, 281)
    seg[seg == 257] = 252
    seg[seg == 258] = 254
    seg[seg == 259] = 256
    seg[seg == 260] = 258
    seg[seg == 261] = 259
    seg[seg == 262] = 260
    seg[seg == 263] = 262
    
    seg[seg == 281] = 251
    seg[seg == 282] = 253
    seg[seg == 283] = 255
    seg[seg == 284] = 257
    seg[seg == 285] = 261
    seg[seg == 286] = 263
    return seg

def postprocessing(parcellated, separated, shift, device):
    """
    Perform post-processing to combine parcellation and hemisphere segmentation results.

    This function fuses the outputs of two neural networks:
      - The *parcellation* network, which labels fine-grained anatomical regions.
      - The *hemisphere* network, which distinguishes left and right hemispheres.

    It uses a predefined mapping (`split_map.pkl`) to merge region and hemisphere
    labels into a unified integer-encoded segmentation map. The output is then
    spatially restored to the original coordinate system using the recorded shift
    and padding offsets.

    Args:
        parcellated (numpy.ndarray): 3D integer array from the parcellation network,
            where each voxel corresponds to an anatomical label (1–142).
        separated (numpy.ndarray): 3D integer array from the hemisphere network,
            where voxel values indicate hemisphere classification:
                0 = background, 1 = left hemisphere, 2 = right hemisphere.
        shift (tuple[int, int, int]): Offsets (xd, yd, zd) used during cropping to
            center the brain; used here to roll the output back to its original location.
        device (torch.device): Device (CPU, CUDA, or MPS) for tensor-based computation.

    Returns:
        numpy.ndarray: The final 3D integer segmentation map where each voxel’s value
        encodes both hemisphere and regional identity, aligned to the original space.
    """
    # -----------------------------------------------------------
    # Step 1: Load the hemisphere–region label correspondence map
    # -----------------------------------------------------------
    # The dictionary in split_map.pkl maps pairs of (hemisphere_label, region_label)
    # to unified segmentation indices.
    with open(SPLIT_MAP_PATH, "rb") as tf:
        dictionary = pickle.load(tf)

    # -----------------------------------------------------------
    # Step 2: Convert input arrays to PyTorch tensors for efficient computation
    # -----------------------------------------------------------
    pmap = torch.tensor(parcellated.astype("int16"), requires_grad=False).to(device)
    hmap = torch.tensor(separated.astype("int16"), requires_grad=False).to(device)

    # Combine flattened hemisphere and parcellation labels into a two-column tensor
    # Each row represents (hemisphere_label, region_label)
    combined = torch.stack((torch.flatten(hmap), torch.flatten(pmap)), axis=-1)

    # Initialize an empty flattened output tensor
    output = torch.zeros_like(hmap).ravel()

    # -----------------------------------------------------------
    # Step 3: Map combined (hemisphere, region) label pairs to final class IDs
    # -----------------------------------------------------------
    # For each entry in the lookup dictionary:
    #   - 'key' represents a pair (hemisphere_label, region_label)
    #   - 'value' is the corresponding unified label in the final segmentation
    for key, value in dictionary.items():
        key = torch.tensor(key, requires_grad=False).to(device)
        mask = torch.all(combined == key, axis=1)
        output[mask] = value

    # Reshape flattened output back to 3D volume
    output = output.reshape(hmap.shape)

    # Move tensor to CPU and convert back to NumPy array
    output = output.cpu().detach().numpy()

    # -----------------------------------------------------------
    # Step 4: Mask irrelevant voxels to clean up final segmentation
    # -----------------------------------------------------------
    # Retain only voxels belonging to hemispheres or specific parcellation indices (87, 138),
    # which likely correspond to midline or reference structures.
    output = output * (np.logical_or(np.logical_or(separated > 0, parcellated == 87), parcellated == 136))

    parc = output.astype("int16")
    hemi = separated.astype("int16")

    # 境界ボクセルを抽出（1と2の隣接）
    boundary = (hemi == 1) & (ndimage.binary_dilation(hemi == 2, structure=np.ones((3,3,3))))
    coords = np.column_stack(np.where(boundary))  # (N,3) 座標
    # 重心
    centroid = coords.mean(axis=0)
    # 共分散行列
    coords_centered = coords - centroid
    cov = np.cov(coords_centered, rowvar=False)
    # 固有分解
    eigvals, eigvecs = np.linalg.eigh(cov)
    # 最小固有値に対応する固有ベクトル = 境界面の法線
    normal = eigvecs[:, np.argmin(eigvals)]
    # 目標法線（X方向）
    target = np.array([1, 0, 0], dtype=float)
    # 回転軸
    v = np.cross(normal, target)
    s = np.linalg.norm(v)
    c = np.dot(normal, target)
    # 外積行列
    vx = np.array([[0, -v[2], v[1]],
                [v[2], 0, -v[0]],
                [-v[1], v[0], 0]])
    # ロドリゲスの回転公式
    R = np.eye(3) + vx + (vx @ vx) * ((1 - c) / (s**2 + 1e-8))

    mask_brain = parc > 0
    coords_brain = np.column_stack(np.where(mask_brain))
    centroid_brain = coords_brain.mean(axis=0)

    # ===== 回転後の parcellation =====
    parc_rot = ndimage.affine_transform(
        parc,
        R,
        offset=centroid - R @ centroid,
        order=0
    )
    parc_rot = ndimage.median_filter(parc_rot, size=3)
    centroid_brain_rot = R @ (centroid_brain - centroid) + centroid
    cx_b, cy_b, cz_b = centroid_brain_rot

    # ===== 扇形（±30度）マスク in 回転後空間 =====
    X, Y, Z = np.meshgrid(
        np.arange(parc_rot.shape[0]),
        np.arange(parc_rot.shape[1]),
        np.arange(parc_rot.shape[2]),
        indexing="ij"
    )

    # 重心からのベクトル（X-Z 平面のみ使用）
    dx = X - cx_b
    dz = Z - cz_b

    r = np.sqrt(dx**2 + dz**2)

    # 半径範囲（例：脳の実体に近い範囲だけ）
    r_min = 5     # 重心近傍を除外（数値不安定対策）
    r_max = 100   # これより外は無視（頭蓋外を除外）
    mask_radius = (r >= r_min) & (r <= r_max)

    # 角度（ラジアン）を計算（-π ~ +π）
    theta = np.arctan2(dx, dz)  # 0° = +Z方向

    # 角度範囲（ラジアン）
    deg = np.deg2rad
    mask_neg30_to_0 = (theta >= -deg(30)) & (theta <= deg(30))
    mask_0_to_pos30 = (theta >= -deg(30)) & (theta <= deg(30))
    mask_neg30_to_0 = mask_neg30_to_0 & mask_radius
    mask_0_to_pos30 = mask_0_to_pos30 & mask_radius

    new_parc_rot = parc_rot.copy()
    # [-30°, 0°]：250 → 275
    mask_A = mask_neg30_to_0 & (parc_rot == 250)
    new_parc_rot[mask_A] = 275
    mask_A = mask_neg30_to_0 & (parc_rot == 252)
    new_parc_rot[mask_A] = 277
    mask_A = mask_neg30_to_0 & (parc_rot == 256)
    new_parc_rot[mask_A] = 279
    # [0°, +30°]：251 → 276
    mask_B = mask_0_to_pos30 & (parc_rot == 251)
    new_parc_rot[mask_B] = 276
    mask_B = mask_0_to_pos30 & (parc_rot == 253)
    new_parc_rot[mask_B] = 278
    mask_B = mask_0_to_pos30 & (parc_rot == 257)
    new_parc_rot[mask_B] = 280

    R_inv = R.T
    new_parc_back = ndimage.affine_transform(
        new_parc_rot,
        R_inv,
        offset=centroid - R_inv @ centroid,
        order=0
    )
    output = relabel_segmentation(new_parc_back)

    # -----------------------------------------------------------
    # Step 5: Restore original spatial position
    # -----------------------------------------------------------
    # Undo the cropping offsets by applying padding and rolling back shifts.
    output = np.pad(output, [(16, 16), (16, 16), (16, 16)], "constant", constant_values=0)
    output = np.roll(output, (-shift[0], -shift[1], -shift[2]), axis=(0, 1, 2))

    # Return the final postprocessed segmentation map
    return output
