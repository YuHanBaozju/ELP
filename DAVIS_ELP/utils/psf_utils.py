import numpy as np

def calculate_rms_radius(psf_img):
    # 获取图像的中心
    h, w = psf_img.shape
    center_x, center_y = w // 2, h // 2

    # 生成坐标网格
    y, x = np.indices((h, w))

    # 计算每个像素到中心的距离
    r_squared = (x - center_x)**2 + (y - center_y)**2

    # 计算加权的距离平方和（用像素强度作为权重）
    weighted_r_squared_sum = np.sum(r_squared * psf_img)

    # 计算总强度（权重总和）
    total_intensity = np.sum(psf_img)

    # 计算 RMS 半径
    rms_radius = np.sqrt(weighted_r_squared_sum / total_intensity)

    return rms_radius