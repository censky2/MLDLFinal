import numpy as np
import cv2
import os

def extract_patches_with_dynamic_step(image, mask, patch_size, step, threshold=0.1, target_classes=None):
    assert image.shape[:2] == mask.shape, "Image and mask must have the same spatial dimensions"

    height, width = image.shape[:2]
    patches_img, patches_mask = [], []
    added_patches = set()
    for y in range(0, height, step):
        for x in range(0, width, step):
            # 提取当前 patch
            patch_img = image[y:y + patch_size, x:x + patch_size]
            patch_mask = mask[y:y + patch_size, x:x + patch_size]
            if patch_img.shape[:2] != (patch_size, patch_size):
                continue
            # 检查目标类别占比
            if target_classes:
                total_pixels = patch_mask.size
                target_pixels = np.sum(np.isin(patch_mask, target_classes))  # 目标类别的像素数量
                target_ratio = target_pixels / total_pixels
            else:
                target_ratio = 0

            # 如果目标类别比例超过阈值
            if target_ratio >= threshold:
                # 保存当前 patch
                patches_img.append(patch_img)
                patches_mask.append(patch_mask)

                # 遍历周围 8个patch
                for dy in [-step//2, 0, step//2]:
                    for dx in [-step//2, 0, step//2]:
                        if dy == 0 and dx == 0:
                            continue
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < height and 0 <= nx < width:
                            # 提取更小的 patch
                            small_patch_img = image[ny:ny + patch_size, nx:nx + patch_size]
                            small_patch_mask = mask[ny:ny + patch_size, nx:nx + patch_size]

                            # 确保 patch 大小正确
                            if small_patch_img.shape[:2] == (patch_size, patch_size):
                                patch_hash = hash(small_patch_img.tobytes())
                                total_pixels = small_patch_mask.size
                                target_pixels = np.sum(np.isin(small_patch_mask, target_classes))  # 目标类别的像素数量
                                target_ratio = target_pixels / total_pixels
                                if patch_hash not in added_patches and target_ratio >= threshold:
                                    print(ny, nx,target_ratio)
                                    added_patches.add(patch_hash)  # 标记为已添加
                                    patches_img.append(small_patch_img)
                                    patches_mask.append(small_patch_mask)

            else:
                # 如果目标类别比例未超过阈值，也保存当前 patch
                patches_img.append(patch_img)
                patches_mask.append(patch_mask)

    return patches_img, patches_mask


IMAGE_FOLDER = r'C:\Users\Shaiiko\Desktop\mldlproject\01_training_dataset_tif_ROIs_processed'
MASK_FOLDER = r'C:\Users\Shaiiko\Desktop\mldlproject\output'
OUTPUT_IMAGE_FOLDER = r'C:\Users\Shaiiko\Desktop\mldlproject\patches\images'
OUTPUT_MASK_FOLDER = r'C:\Users\Shaiiko\Desktop\mldlproject\patches\masks'
PATCH_SIZE = 256
STEP = 128
TARGET_CLASSES = [3, 4, 5]
THRESHOLD = 0.1

# 创建输出文件夹
os.makedirs(OUTPUT_IMAGE_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_MASK_FOLDER, exist_ok=True)
# 示例使用
image_files = sorted(os.listdir(IMAGE_FOLDER))
mask_files = sorted(os.listdir(MASK_FOLDER))

for image_file, mask_file in zip(image_files, mask_files):
    image_path = os.path.join(IMAGE_FOLDER, image_file)
    mask_path = os.path.join(MASK_FOLDER, mask_file)

    # 读取图像和掩码
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)

    # 生成 patch
    image_patches, mask_patches = extract_patches_with_dynamic_step(image, mask, PATCH_SIZE, STEP, THRESHOLD,
                                                                    TARGET_CLASSES)

    # 保存 patch
    for i, (img_patch, mask_patch) in enumerate(zip(image_patches, mask_patches)):
        image_patch_path = os.path.join(OUTPUT_IMAGE_FOLDER, f"{os.path.splitext(image_file)[0]}_patch_{i+10}.tif")
        mask_patch_path = os.path.join(OUTPUT_MASK_FOLDER, f"{os.path.splitext(mask_file)[0]}_patch_{i+10}.tif")

        cv2.imwrite(image_patch_path, img_patch)
        cv2.imwrite(mask_patch_path, mask_patch)

    print(f"处理文件 {image_file} 生成了 {len(image_patches)} 个 patch")
