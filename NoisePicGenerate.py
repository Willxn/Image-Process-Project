import numpy as np
import cv2

# 加载图像
image_path = '/Users/will/Desktop/women.jpeg'  # 替换为你的图像路径
image = cv2.imread(image_path)
rows, cols, channels = image.shape

# 添加高斯噪声
mean = 0
std = 100  # 标准差可以调整来增加或减少噪声强度
gaussian_noise = np.random.normal(mean, std, (rows, cols, channels))
noisy_image_gaussian = image + gaussian_noise
noisy_image_gaussian = np.clip(noisy_image_gaussian, 0, 255).astype(np.uint8)

# 保存高斯噪声图像
cv2.imwrite('/Users/will/Desktop/noisy_image_gaussian.jpg', noisy_image_gaussian)

# 添加椒盐噪声
salt_pepper_ratio = 0.5  # 椒盐比例
amount = 0.09  # 噪声密度
num_salt = np.ceil(amount * image.size * salt_pepper_ratio)
num_pepper = np.ceil(amount * image.size * (1. - salt_pepper_ratio))

noisy_image_sp = np.copy(image)

# 对于椒噪声，随机选择像素点并将其设为白色
for _ in range(int(num_salt)):
    i = np.random.randint(0, rows)
    j = np.random.randint(0, cols)
    noisy_image_sp[i, j] = 255

# 对于盐噪声，随机选择像素点并将其设为黑色
for _ in range(int(num_pepper)):
    i = np.random.randint(0, rows)
    j = np.random.randint(0, cols)
    noisy_image_sp[i, j] = 0

# 保存椒盐噪声图像
cv2.imwrite('/Users/will/Desktop/noisy_image_sp.jpg', noisy_image_sp)

# 显示或保存带噪声图像
cv2.imshow('Gaussian Noise', noisy_image_gaussian)
cv2.imshow('Salt and Pepper Noise', noisy_image_sp)
cv2.waitKey(0)
cv2.destroyAllWindows()
