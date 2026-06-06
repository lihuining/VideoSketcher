import matplotlib.pyplot as plt
import pandas as pd

# def plot_attention_std(file_path):
#     # 读取数据
#     data = pd.read_csv(file_path, header=None, names=['layer', 'type', 'timestep', 'std'])
#
#     # 获取所有独特的层和类型
#     layers = data['layer'].unique()
#     types = data['type'].unique()
#
#     # 创建绘图
#     plt.figure(figsize=(15, 10))
#
#     for layer in layers:
#         for type_ in types:
#             # 选择当前层和类型的数据
#             subset = data[(data['layer'] == layer) & (data['type'] == type_)]
#             plt.plot(subset['timestep'], subset['std'], label=f'{layer}, {type_}')
#
#     # 添加图例和标签
#     plt.xlabel('Timestep')
#     plt.ylabel('Standard Deviation of Attention Map')
#     plt.title('Attention Map Standard Deviation Across Layers and Timesteps')
#     plt.legend()
#     plt.grid(True)
#     plt.savefig('attention_std_plot.png')  # 保存图像
#     # plt.show()  # 显示图像
#
# # 调用函数并传入文件路径
# plot_attention_std('/media/allenyljiang/564AFA804AFA5BE51/Codes/cross-image-attention/outputs/breakdance-flare/ref0020/1.5_2matching_guidance_1start_time51end_time301/attention_std.txt')

# def plot_global_average_attention_std(file_path,save_path):
#     # 读取数据
#     data = pd.read_csv(file_path, header=None, names=['layer', 'type', 'timestep', 'std','mean'])
#
#     # 过滤出 stylized 和 style 数据
#     filtered_data = data[data['type'].isin(['stylized', 'style','struct'])]
#
#     # 计算 stylized 和 style 的全局平均标准差
#     avg_std_stylized = filtered_data[filtered_data['type'] == 'stylized'].groupby('timestep')['std'].mean()
#     avg_std_style = filtered_data[filtered_data['type'] == 'style'].groupby('timestep')['std'].mean()
#     avg_std_struct = filtered_data[filtered_data['type'] == 'struct'].groupby('timestep')['std'].mean()
#
#     # 创建绘图
#     # plt.figure(figsize=(15, 10))
#     fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 15))
#
#
#     # 绘制平均标准差
#     plt.plot(avg_std_stylized.index, avg_std_stylized.values, label='Average Stylized', linestyle='--')
#     plt.plot(avg_std_style.index, avg_std_style.values, label='Average Style', linestyle='-')
#     plt.plot(avg_std_struct.index, avg_std_struct.values, label='Average Struct', linestyle='dashdot')
#     '''
#     supported values are '-', '--', '-.', ':', 'None', ' ', '', 'solid', 'dashed', 'dashdot', 'dotted'
#     '''
#
#     # 添加图例和标签
#     plt.xlabel('Timestep')
#     plt.ylabel('Average Standard Deviation of Attention Map')
#     plt.title('Average Attention Map Standard Deviation Across Timesteps')
#     plt.legend()
#     plt.grid(True)
#     plt.savefig(save_path)  # 保存图像
#     plt.close()
#     # plt.show()  # 显示图像
# def plot_global_average_attention_std(file_path,save_path):
#     # 读取数据
#     data = pd.read_csv(file_path, header=None, names=['layer', 'type', 'timestep', 'std','mean'])
#
#     # 过滤出 stylized 和 style 数据
#     filtered_data = data[data['type'].isin(['stylized', 'style','struct'])]
#
#     # 计算 stylized 和 style 的全局平均标准差
#     avg_std_stylized = filtered_data[filtered_data['type'] == 'stylized'].groupby('timestep')['std'].mean()
#     avg_std_style = filtered_data[filtered_data['type'] == 'style'].groupby('timestep')['std'].mean()
#     avg_std_struct = filtered_data[filtered_data['type'] == 'struct'].groupby('timestep')['std'].mean()
#     #
#     avg_mean_stylized = filtered_data[filtered_data['type'] == 'stylized'].groupby('timestep')['mean'].mean()
#     avg_mean_style = filtered_data[filtered_data['type'] == 'style'].groupby('timestep')['mean'].mean()
#     avg_mean_struct = filtered_data[filtered_data['type'] == 'struct'].groupby('timestep')['mean'].mean()
#
#     # 创建绘图
#     # plt.figure(figsize=(15, 10))
#     fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 15))
#     # 绘制均值曲线
#     ax1.plot(avg_mean_stylized.index, avg_mean_stylized.values, label='Average Stylized Mean', linestyle='--', color='blue')
#     ax1.plot(avg_mean_style.index, avg_mean_style.values, label='Average Style Mean', linestyle='-', color='green')
#     ax1.plot(avg_mean_struct.index, avg_mean_struct.values, label='Average Struct Mean', linestyle='dashdot', color='red')
#
#     # 设置第一个子图的属性
#     ax1.set_xlabel('Time Step')
#     ax1.set_ylabel('Mean Value')
#     ax1.set_title('Mean Over Time')
#     ax1.legend()
#     ax1.grid(True)
#
#     # 绘制标准差曲线
#     ax2.plot(avg_std_stylized.index, avg_std_stylized.values, label='Average Stylized Std', linestyle='--', color='blue')
#     ax2.plot(avg_std_style.index, avg_std_style.values, label='Average Style Std', linestyle='-', color='green')
#     ax2.plot(avg_std_struct.index, avg_std_struct.values, label='Average Struct Std', linestyle='dashdot', color='red')
#
#     # 设置第二个子图的属性
#     ax2.set_xlabel('Time Step')
#     ax2.set_ylabel('Standard Deviation')
#     ax2.set_title('Standard Deviation Over Time')
#     ax2.legend()
#     ax2.grid(True)
#
#     # 保存图形
#     plt.savefig(save_path)
#     plt.close()

def plot_global_average_attention_std(file_path,save_path):
    # 读取数据
    data = pd.read_csv(file_path, header=None, names=['layer', 'type', 'timestep', 'std','mean'])

    # 过滤出 stylized 和 style 数据
    filtered_data = data[data['type'].isin(['stylized', 'style','struct'])]

    # 计算 stylized 和 style 的全局平均标准差
    avg_std_stylized = filtered_data[filtered_data['type'] == 'stylized'].groupby('timestep')['std'].mean()
    avg_std_style = filtered_data[filtered_data['type'] == 'style'].groupby('timestep')['std'].mean()
    avg_std_struct = filtered_data[filtered_data['type'] == 'struct'].groupby('timestep')['std'].mean()
    #
    avg_mean_stylized = filtered_data[filtered_data['type'] == 'stylized'].groupby('timestep')['mean'].mean()
    avg_mean_style = filtered_data[filtered_data['type'] == 'style'].groupby('timestep')['mean'].mean()
    avg_mean_struct = filtered_data[filtered_data['type'] == 'struct'].groupby('timestep')['mean'].mean()

    # 创建绘图
    # plt.figure(figsize=(15, 10))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 15))
    # 绘制均值曲线
    ax1.plot(avg_mean_stylized.index, avg_mean_stylized.values, label='Average Stylized Mean', linestyle='--', color='blue')
    ax1.plot(avg_mean_style.index, avg_mean_style.values, label='Average Style Mean', linestyle='-', color='green')
    ax1.plot(avg_mean_struct.index, avg_mean_struct.values, label='Average Struct Mean', linestyle='dashdot', color='red')

    # 设置第一个子图的属性
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Mean Value')
    ax1.set_title('Mean Over Time')
    ax1.legend()
    ax1.grid(True)

    # 绘制标准差曲线
    ax2.plot(avg_std_stylized.index, avg_std_stylized.values, label='Average Stylized Std', linestyle='--', color='blue')
    ax2.plot(avg_std_style.index, avg_std_style.values, label='Average Style Std', linestyle='-', color='green')
    ax2.plot(avg_std_struct.index, avg_std_struct.values, label='Average Struct Std', linestyle='dashdot', color='red')

    # 设置第二个子图的属性
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Standard Deviation')
    ax2.set_title('Standard Deviation Over Time')
    ax2.legend()
    ax2.grid(True)

    # 保存图形
    plt.savefig(save_path)
    plt.close()
def plot_mean_std_over_time(txt_file, save_path):
    # 读取txt文件
    data = []
    with open(txt_file, 'r') as file:
        for i,line in enumerate(file):
            if i >= 204:
                break
            t, type_, mean, std = line.strip().split(',')
            data.append({
                'time_step': int(t),
                'type': type_.strip(),
                'mean': float(mean.strip()),
                'std': float(std.strip())
            })

    # 转换为DataFrame
    df = pd.DataFrame(data)

    # 按类型分组
    grouped = df.groupby('type')

    # 创建图形和子图
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

    # 绘制均值曲线
    for name, group in grouped:
        ax1.plot(group['time_step'], group['mean'], label=f'{name} Mean', marker='o')

    # 设置第一个子图的属性
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Mean Value')
    ax1.set_title('Mean Over Time')
    ax1.legend()
    ax1.grid(True)

    # 绘制标准差曲线
    for name, group in grouped:
        ax2.plot(group['time_step'], group['std'], label=f'{name} Std', marker='s')

    # 设置第二个子图的属性
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Standard Deviation')
    ax2.set_title('Standard Deviation Over Time')
    ax2.legend()
    ax2.grid(True)

    # 保存图形
    plt.savefig(save_path)
    plt.close()

def plot_adaptive_contrast_over_time_standard(txt_file, png_file):
    # 读取文件
    data = []
    with open(txt_file, 'r') as file:
        for line in file:
            place_in_unet, time_step, contrast_strength,after_contrast_strength = line.strip().split(',')
            data.append({
                'place_in_unet': place_in_unet.strip(),
                'time_step': int(time_step.strip()),
                'contrast_strength': float(contrast_strength.strip()),
                'after_contrast_strength':float(after_contrast_strength.strip())
            })

    # 转换为 DataFrame
    df = pd.DataFrame(data)

    # 计算每个时间步的平均值
    avg_contrast = df.groupby('time_step')['contrast_strength'].mean().reset_index()
    after_avg_contrast = df.groupby('time_step')['after_contrast_strength'].mean().reset_index()

    # 绘制图表
    plt.figure(figsize=(10, 6))
    # plt.plot(avg_contrast['time_step'], avg_contrast['contrast_strength'], marker='o')
    # plt.plot(avg_contrast['time_step'], after_avg_contrast['after_contrast_strength'], marker='s')
    #plt.plot(avg_contrast['time_step'], avg_contrast['contrast_strength'], marker='o', label='Original Contrast Strength')
    plt.plot(avg_contrast['time_step'], after_avg_contrast['after_contrast_strength'], marker='s', label='Contrast Strength')
    plt.title('Sketch Directive Amplification Over Time Steps')
    plt.xlabel('Time Step')
    plt.ylabel('Average Sketch Directive Amplification')
    plt.grid(True)
    plt.legend()  # 显示图例
    plt.savefig(png_file)
    plt.close()
def plot_adaptive_contrast_over_time(txt_file, png_file):
    # 读取文件
    data = []
    with open(txt_file, 'r') as file:
        for line in file:
            place_in_unet, time_step, contrast_strength,after_contrast_strength = line.strip().split(',')
            data.append({
                'place_in_unet': place_in_unet.strip(),
                'time_step': int(time_step.strip()),
                'contrast_strength': float(contrast_strength.strip()),
                'after_contrast_strength':float(after_contrast_strength.strip())
            })

    # 转换为 DataFrame
    df = pd.DataFrame(data)

    # 计算每个时间步的平均值
    avg_contrast = df.groupby('time_step')['contrast_strength'].mean().reset_index()
    after_avg_contrast = df.groupby('time_step')['after_contrast_strength'].mean().reset_index()

    # 绘制图表
    plt.figure(figsize=(10, 6))
    # plt.plot(avg_contrast['time_step'], avg_contrast['contrast_strength'], marker='o')
    # plt.plot(avg_contrast['time_step'], after_avg_contrast['after_contrast_strength'], marker='s')
    plt.plot(avg_contrast['time_step'], avg_contrast['contrast_strength'], marker='o', label='Original Contrast Strength')
    plt.plot(avg_contrast['time_step'], after_avg_contrast['after_contrast_strength'], marker='s', label='After Contrast Strength')
    plt.title('Adaptive Contrast Strength Over Time Steps')
    plt.xlabel('Time Step')
    plt.ylabel('Average Contrast Strength')
    plt.grid(True)
    plt.legend()  # 显示图例
    plt.savefig(png_file)
    plt.close()

if __name__ == '__main__':
    txt_path = "/media/allenyljiang/2CD8318DD83155F4/CVPR2025/Struct_latents/dog/ref0001/2.1_chunk_size2_5/adaptive_contrast.txt"
    #txt_path = "/media/allenyljiang/2CD8318DD83155F4/CVPR2025/Struct_latents/camel/ref0001/2.1_chunk_size2matching_guidance_1start_time231end_time621_1/attention_std.txt"
    save_path = "/media/allenyljiang/2CD8318DD83155F4/CVPR2025/Struct_latents/dog/ref0001/2.1_chunk_size2_5/adaptive_contrast_pa.png"
    # plot_mean_std_over_time(txt_path, save_path)
    # # 调用函数并传入文件路径
    plot_adaptive_contrast_over_time_standard(txt_path,save_path)
