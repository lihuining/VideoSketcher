
# content:/media/allenyljiang/564AFA804AFA5BE51/Codes/StyleID/data/cnt_3doodle/img.png


python run.py \
--app_image_path /media/allenyljiang/564AFA804AFA5BE51/Codes/StyleID/data/style_4sketch_style/4sketch_style1.png \
--struct_image_path /media/allenyljiang/564AFA804AFA5BE51/Codes/Video_Editing/data/tea-pour/000000.png \
--domain_name animal \
--use_masked_adain False \
--contrast_strength 1.67 \
--swap_guidance_scale 1.0 \
--gamma 0.75 \
--prompt "a tea pot pouring tea into a cup."

# #--output_path /media/allenyljiang/564AFA804AFA5BE51/Codes/cross-image-attention/outputs_debug/ \