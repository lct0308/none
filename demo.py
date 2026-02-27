import os
from PIL import Image
import time
import argparse
import cv2

import os
import cv2

def images_to_gif(folder_path, output_gif_path, duration=100):
    # 获取文件夹中的所有文件
    files = os.listdir(folder_path)
    
    # 过滤出图片文件（假设图片文件扩展名为 .jpg, .png, .bmp）
    image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
    
    # 按文件名排序
    image_files.sort()
    
    # 打开第一张图片以获取图像尺寸
    first_image_path = os.path.join(folder_path, image_files[0])
    first_image = Image.open(first_image_path)
    width, height = first_image.size
    
    # 创建一个空的GIF图像列表
    images = []
    
    # 遍历图片文件并添加到GIF图像列表
    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        try:
            # 打开图片
            img = Image.open(image_path)
            
            # 调整图片尺寸以匹配第一张图片的尺寸（可选）
            img = img.resize((width, height), Image.ANTIALIAS)
            
            # 添加到GIF图像列表
            images.append(img)
        except Exception as e:
            print(f"无法打开图片 {image_file}: {e}")
    
    # 保存为GIF文件
    images[0].save(output_gif_path, save_all=True, append_images=images[1:], duration=duration, loop=0)
    print(f"GIF已保存为 {output_gif_path}")

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='demo')
    parser.add_argument("--phase", help="Subdirectory in seq_path.", type=str, default='2023-04-07-21-24-07')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    # all train
    args = parse_args()
    phase_name = args.phase
    folder_path = os.path.join('/data1/pbw/Project/demo', phase_name)

    output_path = f"/data1/pbw/Project/demo/{phase_name}.gif"
    images_to_gif(folder_path, output_path, duration=200)