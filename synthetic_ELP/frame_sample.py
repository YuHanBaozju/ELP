import os
import shutil
import numpy as np


# 定义根目录路径
def frame_sample(root_dir, sample_num):
    convolved_frames_dir = os.path.join(root_dir, 'conv_frames')
    davis_frames_dir = os.path.join(root_dir, f'davis_frames_{sample_num}')

    # 确保目标文件夹存在，不存在则创建
    os.makedirs(davis_frames_dir, exist_ok=True)

    # 如果 davis_frames_dir 里有文件，则删除
    for file in os.listdir(davis_frames_dir):
        file_path = os.path.join(davis_frames_dir, file)
        if os.path.isfile(file_path):
            os.remove(file_path)

    # 获取 convolved_frames 文件夹中所有图像文件并排序
    all_images = sorted(os.listdir(convolved_frames_dir))

    # 特殊情况：当 sample_num 为 1 时，取第 350 张图像
    if sample_num == 1:
        if len(all_images) > 350:
            image = all_images[350]
            src_path = os.path.join(convolved_frames_dir, image)
            dst_path = os.path.join(davis_frames_dir, image)
            shutil.copy(src_path, dst_path)
            # print(f"特殊情况：复制第 350 张图像 {image} 到 davis_frames_1 文件夹中")
        else:
            print("文件夹中的图像不足 350 张，无法完成采样")

    # 普通情况：采样 sample_num 张等距图像
    else:
        total_images = len(all_images)
        if total_images >= sample_num:
            indices = np.linspace(0, total_images - 1, sample_num, dtype=int)
            selected_images = [all_images[i] for i in indices]

            # 复制选择的图像到 davis_frames_{sample_num} 文件夹
            for image in selected_images:
                src_path = os.path.join(convolved_frames_dir, image)
                dst_path = os.path.join(davis_frames_dir, image)
                shutil.copy(src_path, dst_path)
                # print(f"复制 {image} 到 davis_frames_{sample_num} 文件夹中")
        else:
            print(f"文件夹中的图像不足 {sample_num} 张，无法完成等距采样")