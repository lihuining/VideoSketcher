import torch
import numpy as np
from PIL import Image
# def sample_line_points(start, end, num_points=100):
#     """使用线性插值在起点和终点之间生成采样点"""
#     x_values = np.linspace(start[0], end[0], num_points)
#     y_values = np.linspace(start[1], end[1], num_points)
#     return np.vstack((x_values, y_values)).T  # 将 x 和 y 合并成 (num_points, 2) 的形状
def sample_line_points(start, end, num_points=100):
    """使用线性插值在起点和终点之间生成采样点"""
    # 确保 start 和 end 是标量
    start_x, start_y = start[0].item(), start[1].item()
    end_x, end_y = end[0].item(), end[1].item()

    # 使用 torch.linspace 生成线性插值
    x_values = torch.linspace(start_x, end_x, num_points)
    y_values = torch.linspace(start_y, end_y, num_points)
    return torch.stack((x_values, y_values), dim=-1)  # 将 x 和 y 合并成 (num_points, 2) 的形状


def compute_sketch_matching_loss(image1,image2,sparse_matching_lines, sparse_matching_points):
    '''
    numpy:H, W, C W:shape[0] H:shape[1]
    tensor:(B,C, H, W) W:shape[3] H:shape[2]
    input image:tensor
    '''
    if len(image1.shape) == 3:
        image1 = image1.unsqueeze(0)
    if len(image2.shape) == 3:
        image2 = image2.unsqueeze(0)
    weights = torch.tensor([0.299, 0.587, 0.114], device=image1.device,requires_grad=False) # 变为 (3, 1, 1)

    # 计算加权和，保留梯度
    image1 = torch.tensordot(image1, weights, dims=([1], [0])).unsqueeze(1)  # (N, 1, H, W)
    image2 = torch.tensordot(image2, weights, dims=([1], [0])).unsqueeze(1)  # (N, 1, H, W)
    # if isinstance(image1, Image.Image) and isinstance(image2, Image.Image):
    #     image1 = np.array(image1.convert('L'))
    #     image2 = np.array(image2.convert('L'))
    # # 确保输入为 PyTorch 张量
    # image1 = torch.tensor(image1, dtype=torch.float32)
    # image2 = torch.tensor(image2, dtype=torch.float32)
    # 1. 解析 sparse_matching_lines 和 sparse_matching_points
    lines1, lines2 = sparse_matching_lines  # 提取两张图的线段匹配结果
    points1, points2 = sparse_matching_points  # 提取两张图的点匹配结果
    line_loss = torch.tensor(0.0, dtype=image1.dtype).to("cuda")  # 初始化为张量
    # 2. 计算线段匹配误差
    if len(lines1) > 0:  # 确保线段不为空

        for line1, line2 in zip(lines1, lines2):
            # # line1 和 line2 的形状都是 (2, 2)，表示匹配线段的两个端点
            # line_distance = torch.norm(line1 - line2, dim=-1).mean()
            # 采样 line1 和 line2 的点
            line1_points = sample_line_points(line1[0], line1[1]) # (100,2)
            line2_points = sample_line_points(line2[0], line2[1]) # (100,2)
            # 将采样点坐标转化为整数索引，便于在图像中查找像素值
            # line1_points = line1_points.to(torch.int)
            # line2_points = line2_points.to(torch.int)
            line1_points = torch.floor(line1_points).to(torch.int)
            line2_points = torch.floor(line2_points).to(torch.int)

            # 确保采样点在图像范围内
            line1_points = line1_points[(line1_points[:, 0] >= 0) & (line1_points[:, 0] < image1.shape[2]) &
                                        (line1_points[:, 1] >= 0) & (line1_points[:, 1] < image1.shape[3])]
            line2_points = line2_points[(line2_points[:, 0] >= 0) & (line2_points[:, 0] < image2.shape[2]) &
                                        (line2_points[:, 1] >= 0) & (line2_points[:, 1] < image2.shape[3])]
            if len(line1_points) != len(line2_points):
                continue
            # try:
                # 获取每条线的采样点处的像素值
            line1_intensity = image1[0,:,line1_points[:, 1], line1_points[:, 0]]
            line2_intensity = image2[0,:,line2_points[:, 1], line2_points[:, 0]]
            # except:
            #     print("image shape",image2.shape,line2_points[:, 1], line2_points[:, 0])

            # 假设 line1_intensity 和 line2_intensity 是 PyTorch 张量
            line1_intensity = torch.as_tensor(line1_intensity, dtype=torch.float32) # (100,)
            line2_intensity = torch.as_tensor(line2_intensity, dtype=torch.float32)

            # print("line1_intensity requires grad:", line1_intensity.requires_grad)
            # print("line2_intensity requires grad:", line2_intensity.requires_grad)

            # 计算两个采样点集合之间的均方误差损失
            line_loss += torch.norm((line1_intensity - line2_intensity))

            # # 计算两个采样点集合之间的均方误差损失
            # line_loss += np.mean((line1_intensity - line2_intensity) ** 2)

        # 取线段匹配损失的平均
        line_loss /= lines1.shape[0]

    # 3. 计算点匹配误差
    point_loss = torch.tensor(0.0, dtype=image1.dtype).to("cuda")
    if len(points1) > 0:  # 确保点不为空
        for point1, point2 in zip(points1, points2):
            # point1 和 point2 的形状都是 (2,)，表示匹配点的坐标
            # 假设 point1 和 point2 是浮点数列表或数组
            point1 = torch.as_tensor([int(p) for p in point1])
            point2 = torch.as_tensor([int(p) for p in point2])
            # point_distance = torch.norm(image1[point1] - image2[point2])

            # 使用 .item() 提取具体的 x 和 y 坐标
            y1, x1 = point1[1].item(), point1[0].item()  # 注意这里的索引顺序
            y2, x2 = point2[1].item(), point2[0].item()
            # 确保索引在图像范围内
            if 0 <= x1 < image1.shape[2] and 0 <= y1 < image1.shape[3]:
                intensity1 = image1[0,:,y1, x1]  # 从 image1 中获取 (y1, x1) 位置的像素值
            else:
                # print(f"point1 ({x1}, {y1}) 超出了 image1 的范围")
                continue

            if 0 <= x2 < image2.shape[2] and 0 <= y2 < image2.shape[3]:
                intensity2 = image2[0,:,y2, x2]  # 从 image2 中获取 (y2, x2) 位置的像素值
            else:
                # print(f"point2 ({x2}, {y2}) 超出了 image2 的范围")
                continue
            # 计算两个像素点之间的距离
            point_distance = torch.norm(intensity1 - intensity2)
            point_loss += point_distance

        # 取点匹配损失的平均
        point_loss /= points1.shape[0]
        # print("point_loss requires grad:", point_loss.requires_grad)


    # 4. 计算总体匹配损失 (可以加权求和)
    weight = 15 # 计算line loss的时候取的是100个点计算
    total_loss = line_loss + weight*point_loss

    return total_loss,line_loss,weight*point_loss