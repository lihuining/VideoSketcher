from PIL import Image
import numpy as np

# 读取草图并转换为灰度图
#image = Image.open("Dataset/Sketch_dataset/ref2sketch_yr/ref/ref0020.jpg").convert("L")
image = Image.open("Codes/cross-image-attention/outputs/breakdance-flare/ref0020/2_cross_frame/generated_result/0000.png").convert("L")
#image = Image.open("Dataset/Sketch_dataset/ref2sketch_yr/ref/ref0001.jpg").convert("L")
image_array = np.array(image)

# 设置一个阈值（假设小于128的为笔画）
threshold = 128
stroke_pixels = image_array < threshold  # 笔画像素布尔掩码
# 将布尔掩码转换为二值化图像（0 和 255），便于保存
binary_image = (stroke_pixels * 255).astype(np.uint8)

# 转换为PIL图像并保存
binary_image_pil = Image.fromarray(binary_image)
binary_image_pil.save("binary_sketch.png")
# 1. 计算笔画像素占比
stroke_ratio = np.sum(stroke_pixels) / image_array.size
print(f"笔画像素占比：{stroke_ratio:.2%}")

# 2. 计算笔画复杂度（像素变化频率）
# 使用水平和垂直方向的差分来估算像素变化频率
horizontal_diff = np.abs(np.diff(image_array, axis=1))
vertical_diff = np.abs(np.diff(image_array, axis=0))

# 使用给定阈值，计算差分图像中大于阈值的变化像素数量
complexity_threshold = 50
horizontal_complexity = np.sum(horizontal_diff > complexity_threshold)
vertical_complexity = np.sum(vertical_diff > complexity_threshold)

# 总复杂度为水平方向和垂直方向的变化频率之和
complexity = horizontal_complexity + vertical_complexity
print(f"笔画复杂度（像素变化频率）：{complexity}")
