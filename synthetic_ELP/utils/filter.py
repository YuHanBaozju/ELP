def adaptive_filter(data, previous_value, threshold, window_average, alpha=0.5):
    if len(data) < 2:
        return data[-1]

    # 计算当前值与滑动窗口平均值之间的差异
    difference = abs(data[-1] - window_average)

    # 如果差异超过阈值，认为是突变，返回当前值
    if difference > threshold:
        return data[-1]
    else:
        # 否则，进行平滑处理
        return alpha * data[-1] + (1 - alpha) * previous_value

def update_filter_laplacian_product(laplacian_product, filter_laplacian_product, threshold, alpha=0.5, window_size=5):
    if len(laplacian_product) == 1:
        filter_laplacian_product.append(laplacian_product[-1])
        return filter_laplacian_product

    # 计算最后window_size个filter_laplacian_product的平均值
    if len(filter_laplacian_product) < window_size:
        window_average = sum(laplacian_product) / len(laplacian_product)
    else:
        window_average = sum(laplacian_product[-window_size:]) / window_size

    new_value = adaptive_filter(laplacian_product, filter_laplacian_product[-1], threshold, window_average, alpha)
    filter_laplacian_product.append(new_value)

    return filter_laplacian_product
