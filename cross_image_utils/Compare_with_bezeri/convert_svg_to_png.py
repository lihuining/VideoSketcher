import cairosvg
from PIL import Image

svg_path = '/media/allenyljiang/564AFA804AFA5BE5/Codes/StyleID/data/style_2d_sketch_svg/camel_16strokes_seed0_best.svg'
png_path = '/media/allenyljiang/564AFA804AFA5BE5/Codes/StyleID/data/style_2d_sketch/camel_16strokes_seed0_best_rgb.png'
# 读取SVG文件并转换为PNG格式
with open(svg_path, 'rb') as svg_file:
    svg_data = svg_file.read()
    png_data = cairosvg.svg2png(bytestring=svg_data)

# 创建一个白色背景的新图像
background = Image.new("RGB", png_data.size, (255, 255, 255))

# 将RGBA图像的内容粘贴到背景上，并忽略透明度
background.paste(png_data, mask=png_data.split()[3])  # 使用Alpha通道作为掩码

# 保存为新的文件
background.save(png_path)

from PIL import Image

# 读取转换后的PNG文件
with Image.open(png_path) as img:
    print(img.mode)  # 输出图像模式，如 "RGB" 或 "L"
    print(img.getextrema())  # 输出像素值的最小和最大值