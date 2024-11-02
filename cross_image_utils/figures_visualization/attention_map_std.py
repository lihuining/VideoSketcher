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
import matplotlib.pyplot as plt
import pandas as pd

def plot_global_average_attention_std(file_path,save_path):
    # 读取数据
    data = pd.read_csv(file_path, header=None, names=['layer', 'type', 'timestep', 'std'])

    # 过滤出 stylized 和 style 数据
    filtered_data = data[data['type'].isin(['stylized', 'style','struct'])]

    # 计算 stylized 和 style 的全局平均标准差
    avg_std_stylized = filtered_data[filtered_data['type'] == 'stylized'].groupby('timestep')['std'].mean()
    avg_std_style = filtered_data[filtered_data['type'] == 'style'].groupby('timestep')['std'].mean()
    avg_std_struct = filtered_data[filtered_data['type'] == 'struct'].groupby('timestep')['std'].mean()

    # 创建绘图
    plt.figure(figsize=(15, 10))

    # 绘制平均标准差
    plt.plot(avg_std_stylized.index, avg_std_stylized.values, label='Average Stylized', linestyle='--')
    plt.plot(avg_std_style.index, avg_std_style.values, label='Average Style', linestyle='-')
    plt.plot(avg_std_struct.index, avg_std_struct.values, label='Average Struct', linestyle='dashdot')
    '''
    supported values are '-', '--', '-.', ':', 'None', ' ', '', 'solid', 'dashed', 'dashdot', 'dotted'
    '''

    # 添加图例和标签
    plt.xlabel('Timestep')
    plt.ylabel('Average Standard Deviation of Attention Map')
    plt.title('Average Attention Map Standard Deviation Across Timesteps')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)  # 保存图像
    # plt.show()  # 显示图像
if __name__ == '__main__':

    # 调用函数并传入文件路径
    plot_global_average_attention_std('/media/allenyljiang/564AFA804AFA5BE51/Codes/cross-image-attention/outputs/breakdance-flare/ref0020/1.5_2matching_guidance_1start_time51end_time301/attention_std.txt')
