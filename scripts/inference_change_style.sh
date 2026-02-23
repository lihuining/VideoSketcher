
#!/bin/bash

# YAML 文件路径
yaml_file="Codes/cross-image-attention/configs/wild-motor.yaml"
cd "Codes/cross-image-attention"

# 读取 struct_video_list.txt 文件中的所有视频路径
while IFS= read -r app_image_path; do
    # 检查路径是否为空
#    if [ -n "$input_path" ]; then
#        # 使用 sed 修改 YAML 文件中的 input_path 参数
#        sed -i "s|^input_path:.*|input_path: \"$input_path\"|" "$yaml_file"
#        echo "Processing video: $input_path"
    if [ -n "$app_image_path" ]; then
          # 使用 sed 修改 YAML 文件中的 app_image_path 参数
          sed -i "s|^app_image_path:.*|app_image_path: \"$app_image_path\"|" "$yaml_file"
          echo "Processing style image: $app_image_path"

        # 运行 Python 脚本，使用更新后的配置文件
        python3 video_appearance_transfer_model.py --config ${yaml_file}
    fi
done < "Codes/cross-image-attention/experiments/rebuttal/struct/struct_video_final.txt"
#将 struct_video_list.txt 文件的内容作为 while 循环的标准输入。

