# import os.path
#
# import matplotlib.pyplot as plt
# import pandas as pd
#
# # 读取并解析数据
# data = []
# txt_data_path = "/media/allenyljiang/2CD8318DD83155F4/CVPR2025/Struct_latents/goat/ref0020/2.1_chunk_size2/attention_std.txt"
# #txt_data_path = "/media/allenyljiang/2CD8318DD83155F4/CVPR2025/Struct_latents/blackswan/ref0030/keep_struct_end/2.1_chunk_size2__1/attention_std.txt"
# with open(txt_data_path, "r") as file:
#     for line in file:
#         parts = line.strip().split(",")
#         level, type_, timestep, value = parts[0], parts[1], int(parts[2]), float(parts[3])
#         data.append((level, type_, timestep, value))
#
# # 转换为 DataFrame
# df = pd.DataFrame(data, columns=["level", "type", "timestep", "value"])
# save_dir = "/media/allenyljiang/564AFA804AFA5BE51/Codes/cross-image-attention/cross_image_utils/figures_visualization/goat_each_step"
# # 获取所有的时间步
# timesteps = sorted(df["timestep"].unique())
#
# # 为每个时间步绘制图像
# for timestep in timesteps:
#     # 筛选出当前时间步的数据
#     df_timestep = df[df["timestep"] == timestep]
#     df_pivot = df_timestep.pivot(index="level", columns="type", values="value")
#     # 初始化图像
#     plt.figure(figsize=(10, 6))
#
#     # 为每个类型绘制曲线
#     # for type_ in ["style", "stylized", "struct"]:
#     #     df_type = df_timestep[df_timestep["type"] == type_]
#     #     plt.plot(df_type["level"], df_type["value"], label=type_)
#     # 为每个类型绘制曲线
#     for type_ in ["style", "stylized", "struct"]:
#         if type_ in df_pivot.columns:
#             plt.plot(df_pivot.index, df_pivot[type_], label=type_)
#
#     # 添加图例和标签
#     plt.xlabel("Level (from down_1_self to up_17_self)")
#     plt.ylabel("Value")
#     plt.title(f"Trend at Timestep {timestep}")
#     plt.legend()
#     plt.xticks(rotation=45)
#     plt.grid()
#     plt.tight_layout()
#
#     # 保存图像
#     plt.savefig(os.path.join(save_dir,f"timestep_{timestep}.png"))
#     plt.close()

import os.path
import matplotlib.pyplot as plt
import pandas as pd

# 读取并解析数据
data = []
txt_data_path = "/home/allenyljiang/Desktop/CVPR25/Struct_latents/car-turn/ref0030/2.1_chunk_size2/attention_std.txt"
save_dir = "/media/allenyljiang/564AFA804AFA5BE51/Codes/cross-image-attention/cross_image_utils/figures_visualization/car-turn"
# txt_data_path = "/media/allenyljiang/2CD8318DD83155F4/CVPR2025/Struct_latents/goat/ref0020/2.1_chunk_size2/attention_std.txt"
# txt_data_path = "/media/allenyljiang/2CD8318DD83155F4/CVPR2025/Struct_latents/blackswan/ref0030/keep_struct_end/2.1_chunk_size2__1/attention_std.txt"
os.makedirs(save_dir,exist_ok=True)
with open(txt_data_path, "r") as file:
    for line in file:
        parts = line.strip().split(",")
        if len(parts) != 4:
            continue  # 跳过格式不正确的行
        level, type_, timestep, value = parts[0], parts[1], int(parts[2]), float(parts[3])
        data.append((level, type_, timestep, value))

# 转换为 DataFrame
df = pd.DataFrame(data, columns=["level", "type", "timestep", "value"])

# 定义保存目录


# 获取所有的时间步
timesteps = sorted(df["timestep"].unique())

# 自定义排序函数
def custom_sort_key(level):
    prefix, number,_ = level.split("_")
    prefix_order = {"down": 0, "mid": 1, "up": 2}
    return (prefix_order[prefix], int(number))

# 为每个时间步绘制图像
for timestep in timesteps:
    # 筛选出当前时间步的数据
    df_timestep = df[df["timestep"] == timestep]

    # 对数据进行聚合，按 level 和 type 分组，取 value 的平均值
    df_grouped = df_timestep.groupby(["level", "type"]).mean().reset_index()

    # 使用 pivot 重新组织数据
    df_pivot = df_grouped.pivot(index="level", columns="type", values="value")
    # 按自定义排序规则排序
    df_pivot = df_pivot.loc[sorted(df_pivot.index, key=custom_sort_key)]
    # 初始化图像
    plt.figure(figsize=(10, 6))

    # 为每个类型绘制曲线
    for type_ in ["style", "stylized", "struct"]:
        if type_ in df_pivot.columns:
            plt.plot(df_pivot.index, df_pivot[type_], label=type_)

    # 添加图例和标签
    plt.xlabel("Level (from down_1_self to up_17_self)")
    plt.ylabel("Value")
    plt.title(f"Trend at Timestep {timestep}")
    plt.legend()
    plt.xticks(rotation=45)
    plt.grid()
    plt.tight_layout()

    # 保存图像
    plt.savefig(os.path.join(save_dir, f"timestep_{timestep}.png"))
    plt.close()
