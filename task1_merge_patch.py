import os
import cv2
import matplotlib.pyplot as plt
import numpy as np


def reconstruct_from_patches(patches, shape_info, patch_size, stride):
    reconstructed = np.zeros((1024,1024), dtype=np.float32)
    count_map = np.zeros((1024,1024), dtype=np.float32)

    idx = 0
    for i in range(0, 1024 - patch_size + 1, stride):
        for j in range(0, 1024 - patch_size + 1, stride):
            reconstructed[i:i + patch_size, j:j + patch_size] += patches[idx]
            count_map[i:i + patch_size, j:j + patch_size] += 1
            idx += 1

    return (reconstructed / np.maximum(count_map, 1)).astype(np.uint8)


def load_patches(patch_folder, image_name):
    print(image_name)
    patches = []
    for filename in os.listdir(patch_folder):

        if filename.startswith(image_name) and "patch" in filename and filename.endswith(".tif"):
            patch = cv2.imread(os.path.join(patch_folder, filename), cv2.IMREAD_UNCHANGED)
            patches.append(patch)
    return np.array(patches)


def merge_predictions_and_masks(patch_folder, mask_patch_folder, output_folder, shape_info_path, patch_size, stride):
    os.makedirs(output_folder, exist_ok=True)

    shape_info_dict = np.load(shape_info_path, allow_pickle=True).item()

    for image_name in shape_info_dict:

        shape_info = shape_info_dict[image_name]
        # 合并图像
        # patches = load_patches(patch_folder, os.path.splitext(image_name)[0])
        # reconstructed_image = reconstruct_from_patches(patches, shape_info, patch_size, stride)

        # 合并 mask
        mask_patches = load_patches(mask_patch_folder, os.path.splitext(image_name))
        reconstructed_mask = reconstruct_from_patches(mask_patches, shape_info, patch_size, stride)

        # 保存
        # output_image_path = os.path.join(output_folder, image_name)
        output_mask_path = os.path.join(output_folder, image_name)

        # cv2.imwrite(output_image_path, reconstructed_image)
        cv2.imwrite(output_mask_path, reconstructed_mask)
        print(f"Merged image and mask saved for {image_name}")
        exit(0)


# 示例用法
if __name__ == "__main__":
    patch_folder = "output_image_patches"  # 预测的图像 patch 文件夹
    mask_patch_folder = "output_mask_patches"  # 预测的 mask patch 文件夹
    output_folder = "merged_results"  # 合并结果输出文件夹
    shape_info_path = "output_image_patches/shape_info.npy"  # shape 信息路径

    patch_size = 256
    stride = 128

    merge_predictions_and_masks(patch_folder, mask_patch_folder, output_folder, shape_info_path, patch_size, stride)
