import os
from PIL import Image
import cairosvg
import io

def convert_svg_to_png(svg_path, png_path):
    # 读取SVG文件并转换为PNG格式
    with open(svg_path, 'rb') as svg_file:
        svg_data = svg_file.read()
        png_data = cairosvg.svg2png(bytestring=svg_data)

    # 将PNG数据加载为PIL图像
    png_image = Image.open(io.BytesIO(png_data))

    # 创建一个白色背景的新图像
    background = Image.new("RGB", png_image.size, (255, 255, 255))

    # 将RGBA图像的内容粘贴到背景上，并忽略透明度
    background.paste(png_image, mask=png_image.split()[3])  # 使用Alpha通道作为掩码

    # 保存为新的文件
    background.save(png_path)

def process_svg_files_in_folder(svg_folder, png_folder):
    # 遍历文件夹中的所有SVG文件
    for filename in os.listdir(svg_folder):
        if filename.endswith('.svg'):
            svg_path = os.path.join(svg_folder, filename)
            png_path = os.path.join(png_folder, os.path.splitext(filename)[0] + '.png')
            convert_svg_to_png(svg_path, png_path)
            print(f"Converted {svg_path} to {png_path}")

def process_svg_files(txt_file_path):
    # 读取文本文件中的每一行
    with open(txt_file_path, 'r') as file:
        lines = file.readlines()

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # 获取SVG文件夹路径
        svg_folder = line
        png_folder = svg_folder + '_png'

        # 创建目标文件夹
        os.makedirs(png_folder, exist_ok=True)

        # 处理文件夹中的所有SVG文件
        process_svg_files_in_folder(svg_folder, png_folder)

if __name__ == "__main__":
    txt_file_path = '/media/allenyljiang/564AFA804AFA5BE51/Codes/cross-image-attention/experiments/sketch_video_synthesis.txt'
    process_svg_files(txt_file_path)
