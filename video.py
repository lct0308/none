import cv2
import os

ori_path = '/data1/pbw/Project/visualization'

# 读取图片路径的文本文件
input_txt = '/data1/pbw/Project/test_seq/demo.txt'
output_video = 'lane_det.mp4'
fps = 10  # 视频帧率

# 读取图片路径
with open(input_txt, 'r') as file:
    image_paths = file.readlines()

# 去除路径中的换行符
image_paths = [path.strip() for path in image_paths]

# 检查是否有图片路径
if not image_paths:
    print("文本文件中没有图片路径。")
    exit()

# 读取第一张图片以获取视频的分辨率
first_image = cv2.imread(os.path.join(ori_path, image_paths[0]))
height, width, layers = first_image.shape

# 创建视频编写器
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 视频编码格式
video = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

# 按顺序将图片写入视频
for image_path in image_paths:
    img = cv2.imread(os.path.join(ori_path, image_path))
    if img is None:
        print(f"无法读取图片: {image_path}")
        continue
    video.write(img)

# 释放视频编写器
video.release()

print(f"视频已生成: {output_video}")