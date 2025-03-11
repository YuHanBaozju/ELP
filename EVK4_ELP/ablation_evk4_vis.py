import glob
import time

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
from utils.filter import update_filter_laplacian_product




# 初始化全局变量
roi_start = None
roi_end = None
drawing = False
img = None
img_copy = None
ROI = None  # 用于存储ROI区域

# 鼠标回调函数，用于处理鼠标事件
# 鼠标回调函数，用于处理鼠标事件
def mouse_callback(event, x, y, flags, param):
    global roi_start, roi_end, drawing, img_copy, ROI

    # 当左键按下时，开始绘制 ROI
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        roi_start = (x, y)
        roi_end = (x, y)  # 初始化结束坐标与开始坐标相同

    # 当鼠标移动时，更新 ROI 结束位置
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            roi_end = (x, y)
            img_copy = img.copy()  # 在临时图像上绘制矩形框
            cv2.rectangle(img_copy, roi_start, roi_end, (0, 255, 0), 2)
            cv2.imshow("EvTemMap", img_copy)

    # 当左键释放时，完成 ROI 的绘制并关闭窗口
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        roi_end = (x, y)
        cv2.rectangle(img_copy, roi_start, roi_end, (0, 255, 0), 2)
        cv2.imshow("EvTemMap", img_copy)

        # 确定ROI的四个边界值
        x1, y1 = min(roi_start[0], roi_end[0]), min(roi_start[1], roi_end[1])
        x2, y2 = max(roi_start[0], roi_end[0]), max(roi_start[1], roi_end[1])

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(img.shape[1], x2)
        y2 = min(img.shape[0], y2)

        ROI = (y1, y2,x1, x2)  # 存储 ROI 的四个边界值

        # 输出最终选择的 ROI 区域
        print(f"ROI selected: {ROI}")

        # 关闭窗口
        cv2.destroyAllWindows()

def render(image, x, y, p):
    """
    make event image
    """
    height, width = image.shape[:2]
    x = np.clip(x, 0, width - 1)
    y = np.clip(y, 0, height - 1)
    image[y[p == 1.], x[p == 1.], 2] = 255
    image[y[p == 0.], x[p == 0.], 0] = 255

    return image

def event_acc(events, shape):
    # 创建 p_events 和 n_events 数组
    p_events = np.zeros(shape, dtype='uint8')
    n_events = np.zeros(shape, dtype='uint8')

    # 将正负事件分开处理
    pos_events = events[events['p'] == 1]
    neg_events = events[events['p'] == 0]

    # 分别对正负事件进行累加

    np.add.at(p_events, (pos_events['y'], pos_events['x']), 1)
    np.add.at(n_events, (neg_events['y'], neg_events['x']), 1)

    return p_events, n_events




def elp_ablation_evk4(root_folder,vis=False, abalation_on='laplacian'):
    frame_folder = root_folder + '/open/pic'

    import re

    # 打开并读取文件内容
    with open(root_folder + '/info.txt', 'r') as file:
        data = file.read()

    # 使用正则表达式查找gt_focus的值
    gt_focus_pattern = r'gt_timestamp\s*=\s*(\d+)'
    gt_focus_match = re.search(gt_focus_pattern, data)
    if gt_focus_match:
        gt_focus_value = int(gt_focus_match.group(1))
        # print(f"gt_focus: {gt_focus_value}")

    # 使用正则表达式查找ROI selected的值
    roi_pattern = r'ROI selected:\s*\(\s*(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\s*\)'
    roi_match = re.search(roi_pattern, data)

    # 使用正则表达式查找delta_t的值
    delta_t_pattern = r'delta_t\s*=\s*([\d.]+)'
    delta_t_match = re.search(delta_t_pattern, data)
    if delta_t_match:
        delta_t = float(delta_t_match.group(1))
        # print(f"delta_t: {delta_t}")
    else:
        delta_t = 1e3
        # print(f"delta_t: {delta_t}")

    event = np.load(root_folder + '/event.npy')

    start_time = event['t'][0]
    if abalation_on == 'filter':
        start_time = event['t'][0] + 50e3

    end_time = event['t'][-1]

    # roi = [0, 100,0, 100]
    # roi = [150, 250,150,250]

    jpg_files = glob.glob(os.path.join(frame_folder, '*.png'))

    laplacian_product = []

    filter_laplacian_product = []
    t_list = []

    find_focus = False

    img_path = jpg_files[0]
    img = cv2.imread(img_path)

    if img is None:
        print("Failed to load the image.")
    else:
        if roi_match:
            ROI = tuple(map(int, roi_match.groups()))
            # print(f"ROI selected: {ROI}")
        else:

            img_copy = img.copy()  # 创建一个副本用于交互
            cv2.namedWindow("EvTemMap")
            cv2.setMouseCallback("EvTemMap", mouse_callback)

            # 显示图像，等待用户选择 ROI
            while True:
                cv2.imshow("EvTemMap", img_copy)
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # 按下 'ESC' 退出程序（备选项）
                    break

    runtime = 0

    for i, file in enumerate(jpg_files):

        img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        img = cv2.medianBlur(img, 3)
        # img = cv2.GaussianBlur(img, (11, 11), 5)
        # img_roi = img[roi[0]:roi[1], roi[2]:roi[3]]
        if abalation_on == 'filter':
            laplacian = cv2.Laplacian(img, cv2.CV_32F)

        if abalation_on == 'laplacian':
            laplacian = np.ones(img.shape, dtype='float32')

        # laplacian_mask = np.zeros_like(laplacian)
        # laplacian_mask[ROI[0]:ROI[1], ROI[2]:ROI[3]] = laplacian[ROI[0]:ROI[1], ROI[2]:ROI[3]]
        laplacian = laplacian[ROI[0]:ROI[1], ROI[2]:ROI[3]]

        # plt.imshow(laplacian)
        # plt.show()

        if vis:
            img_show = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            img_show = cv2.rectangle(img_show, (ROI[2], ROI[0]), (ROI[3], ROI[1]), (0, 255, 0), 2)
            cv2.imshow('show', img_show)
            cv2.waitKey(0)

        event_all = event[
            (event['y'] >= ROI[0]) & (event['y'] < ROI[1]) & (event['x'] >= ROI[2]) & (event['x'] < ROI[3])]
        event_all['x'] = event_all['x'] - ROI[2]
        event_all['y'] = event_all['y'] - ROI[0]

        t = start_time

        while t < end_time:
            if abalation_on == 'filter':
                if find_focus:
                    break

            # mask_start = event_all['t'] >= t
            # mask_end = event_all['t'] <= t + delta_t
            #
            # # 合并条件后的索引
            # event_sample = event_all[mask_start & mask_end]

            # 假设 event_all['t'] 已排序
            start_idx = np.searchsorted(event_all['t'], t, side='left')
            end_idx = np.searchsorted(event_all['t'], t + delta_t, side='right')

            # 使用切片获取时间范围内的事件
            event_sample = event_all[start_idx:end_idx]
            # event_sample = event_all[(event_all['t'] >= t) & (event_all['t'] <= t+delta_t)]

            if len(event_sample) < 10:
                t += delta_t
                continue

            runtime_start = time.time()
            p_events, n_events = event_acc(event_sample, laplacian.shape)

            laplacian_product_index = -np.sum(laplacian * p_events) + np.sum(laplacian * n_events)
            t_list.append(t)
            laplacian_product.append(laplacian_product_index)


            runtime += time.time() - runtime_start

            threshold = 5e4  # 设置一个阈值来检测突变
            alpha = 0.4  # 平滑系数
            filter_size = 10



            # for ablation
            if abalation_on == 'filter':
                filter_laplacian_product = laplacian_product
            else:
                filter_laplacian_product = update_filter_laplacian_product(laplacian_product, filter_laplacian_product,
                                                                           threshold,
                                                                           alpha, window_size=filter_size)

            # event_frame = np.zeros((img.shape[0], img.shape[1], 3), dtype='uint8')
            # event_frame = render(event_frame, event_sample['x'], event_sample['y'], event_sample['p'])

            t += delta_t

            # if min(filter_laplacian_product[-5:]) < -200 and len(filter_laplacian_product) > 5 and max(
            #         filter_laplacian_product[:-5]) > 50:

            if min(filter_laplacian_product[-5:]) < -200 and len(filter_laplacian_product) > 5 and max(
                    filter_laplacian_product[:-5]) > 50:
                # 将列表转换为 NumPy 数组
                arr = np.array(filter_laplacian_product)

                # 找到从正数跳到负数的位置，判断前一个元素为正，当前元素为负
                transitions = np.where((arr[:-1] > 0) & (arr[1:] < 0))[0] + 1

                focus_timestamp = t_list[transitions[-1]] - delta_t * 2

                find_focus = True

            # cv2.imshow('show', add)
            # cv2.waitKey(1)

    print(f'error: {focus_timestamp - gt_focus_value}')
    cv2.destroyAllWindows()

    if vis:
        matplotlib.rcParams['font.family'] = 'Times New Roman'
        plt.plot([x / 1e3 for x in t_list], [x / 3e2 for x in laplacian_product], color='black', label='ELP')
        plt.plot([x / 1e3 for x in t_list], [x / 3e2 for x in filter_laplacian_product], color='red', linestyle='--',
                 label='Filtered ELP')
        plt.axvline(x=focus_timestamp / 1e3, color='blue', linestyle=':', label='Motor stop time')
        plt.axvline(x=gt_focus_value / 1e3, color='green', linestyle='-', label='GT focus timestamp')

        plt.xlabel('Time (ms)', fontsize=20)
        plt.ylabel('ELP', fontsize=20)

        plt.legend(fontsize=18)
        # 使用 tick_params 设置坐标轴刻度数字的字体大小
        plt.tick_params(axis='both', which='major', labelsize=18)  # major 刻度数字
        plt.tick_params(axis='both', which='minor', labelsize=18)  # minor 刻度数字（如果有）

        plt.show()

# 静止场景
# root_folder = r'E:\Event_camera\DVS_AF_est\dataset\EVK4\box\bright_motion'
# elp_ablation_evk4(root_folder,vis=True,abalation_on='filter')




