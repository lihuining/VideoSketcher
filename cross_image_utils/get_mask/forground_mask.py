import cv2
import numpy as np

# 读取图像
image = cv2.imread('/media/allenyljiang/5234E69834E67DFB/Dataset/Sketch_dataset/ref2sketch_yr/ref/ref0020.jpg', cv2.IMREAD_GRAYSCALE)

# 使用自适应阈值处理来生成二值化的前景 mask
# 参数说明：
# - maxValue: 二值化后前景的像素值（通常为255）
# - adaptiveMethod: 使用均值或高斯的方式计算局部阈值（cv2.ADAPTIVE_THRESH_MEAN_C 或 cv2.ADAPTIVE_THRESH_GAUSSIAN_C）
# - thresholdType: 二值化类型，通常为 cv2.THRESH_BINARY
# - blockSize: 计算局部阈值时的邻域大小（一般使用奇数）
# - C: 常数C，用于调节阈值结果
foreground_mask = cv2.adaptiveThreshold(image, maxValue=255,
                                        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        thresholdType=cv2.THRESH_BINARY,
                                        blockSize=11,
                                        C=2)
cv2.imwrite('/media/allenyljiang/564AFA804AFA5BE51/Codes/cross-image-attention/debug/masks/forground.png',foreground_mask)


# # 显示结果
# cv2.imshow('Foreground Mask', foreground_mask)
# cv2.waitKey(0)
# cv2.destroyAllWindows()