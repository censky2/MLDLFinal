import os
import cv2
import numpy as np
from skimage.util import view_as_windows


def split_into_patches(image, patch_size, stride,status):
    if status=='mask':
        window_shape = (patch_size, patch_size)
        patches = view_as_windows(image, window_shape, step=stride)
        return patches.reshape(-1, patch_size, patch_size), patches.shape
    else:
        window_shape = (patch_size, patch_size,3)
        patches = view_as_windows(image, window_shape, step=stride)
        return patches.reshape(-1, patch_size, patch_size,3), patches.shape


def save_patches(image_path, mask_path, patch_size, stride, output_image_folder, output_mask_folder):
    # 读取图像和对应的 mask
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)

    image_name = os.path.splitext(os.path.basename(image_path))[0]

    # 图像和 mask 分别切分为 patches
    image_patches, shape_info = split_into_patches(image, patch_size, stride,'image')
    mask_patches, _ = split_into_patches(mask, patch_size, stride,'mask')

    # 保存每个 patch
    for idx, (img_patch, mask_patch) in enumerate(zip(image_patches, mask_patches)):
        img_patch_filename = f"{image_name}_patch_{idx+10}.tif"
        mask_patch_filename = f"{image_name}_patch_{idx+10}.tif"

        cv2.imwrite(os.path.join(output_image_folder, img_patch_filename), img_patch)
        cv2.imwrite(os.path.join(output_mask_folder, mask_patch_filename), mask_patch)

    return shape_info


def process_image_and_mask_folders(input_image_folder, input_mask_folder, output_image_folder, output_mask_folder,
                                   patch_size, stride):
    os.makedirs(output_image_folder, exist_ok=True)
    os.makedirs(output_mask_folder, exist_ok=True)

    shape_info_dict = {}
    for filename in os.listdir(input_image_folder):
        if filename.endswith(".tif"):
            image_path = os.path.join(input_image_folder, filename)
            mask_path = os.path.join(input_mask_folder, filename)  # 假设图像和 mask 文件名一致

            if not os.path.exists(mask_path):
                print(f"Mask file for {filename} not found, skipping...")
                continue

            shape_info = save_patches(image_path, mask_path, patch_size, stride, output_image_folder,
                                      output_mask_folder)
            shape_info_dict[filename] = shape_info

    # 保存形状信息以备合并时使用
    np.save(os.path.join(output_image_folder, "shape_info.npy"), shape_info_dict)
    print("All images and masks processed and shape info saved.")


# 示例用法
if __name__ == "__main__":
    input_image_folder = r'C:\Users\Shaiiko\Desktop\mldlproject\01_training_dataset_tif_ROIs_processed'  # 输入图像文件夹
    input_mask_folder = r'C:\Users\Shaiiko\Desktop\mldlproject\output'  # 输入 mask 文件夹
    output_image_folder = "output_image_patches"  # 输出图像 patch 文件夹
    output_mask_folder = "output_mask_patches"  # 输出 mask patch 文件夹

    patch_size = 256  # patch 的大小
    stride = 128  # patch 的步长

    process_image_and_mask_folders(input_image_folder, input_mask_folder, output_image_folder, output_mask_folder,
                                   patch_size, stride)
