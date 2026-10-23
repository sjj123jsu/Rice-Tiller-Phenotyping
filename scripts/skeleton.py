
import os
import cv2
import numpy as np
from skimage.morphology import skeletonize


def filter_mask_by_area(mask, min_area=300):
    """根据面积筛选，去掉小噪声区域"""
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    filtered = np.zeros_like(mask)
    for i in range(1, num_labels):  # 从1开始跳过背景
        if stats[i, cv2.CC_STAT_AREA] > min_area:
            filtered[labels == i] = 255
    return filtered


def find_skeleton(mask):
    """提取单像素骨架"""
    return skeletonize(mask // 255).astype(np.uint8) * 255


def process_recursive(root_input, root_output):
    """遍历主文件夹下的所有子文件夹并处理"""
    # 获取主文件夹下所有内容
    subfolders = [f for f in os.listdir(root_input) if os.path.isdir(os.path.join(root_input, f))]

    if not subfolders:
        print(f"⚠️ 在 {root_input} 中未发现子文件夹，请检查路径。")
        return

    for subfolder in subfolders:
        input_sub_path = os.path.join(root_input, subfolder)
        output_sub_path = os.path.join(root_output, subfolder)  # 输出文件夹命名规则

        os.makedirs(output_sub_path, exist_ok=True)
        print(f"📂 正在处理文件夹: {subfolder}...")

        # 遍历子文件夹内的图片
        for filename in os.listdir(input_sub_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                path = os.path.join(input_sub_path, filename)
                mask = cv2.imread(path, 0)
                if mask is None:
                    continue

                # 1. 阈值化处理
                _, binary = cv2.threshold(mask, 37, 255, cv2.THRESH_BINARY)

                # 2. 连通域面积过滤（抑制孤立噪声）
                filtered = filter_mask_by_area(binary)

                # 3. 提取拓扑骨架
                skeleton = find_skeleton(filtered)

                # 保存结果
                out_path = os.path.join(output_sub_path, filename)
                cv2.imwrite(out_path, skeleton)

        print(f"   --- {subfolder} 处理完成！")

    print("\n✅ 所有子文件夹的骨架图提取任务已全部完成！")


# ✅ 使用方法
if __name__ == '__main__':
    # 这里填写包含多个水稻视角序列文件夹的总目录
    input_root_folder = r"D:\lunwenyong\VID_20250816_174725mask"
    # 这里填写你希望保存结果的总目录
    output_root_folder = r"D:\lunwenyong\VID_20250816_174725maskgujia"

    process_recursive(input_root_folder, output_root_folder)