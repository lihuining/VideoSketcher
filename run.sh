
# content:/media/allenyljiang/564AFA804AFA5BE51/Codes/StyleID/data/cnt_3doodle/img.png


python run.py \
--app_image_path /media/allenyljiang/564AFA804AFA5BE51/Codes/mixsa/data/style_sketch/ref_sample3.jpg \
--struct_image_path /media/allenyljiang/564AFA804AFA5BE51/Codes/mixsa/data/ref2sketch_dataset/Em8CEBGVcAEGlt6.jfif \
--domain_name animal \
--use_masked_adain True \
--contrast_strength 1.67 \
--swap_guidance_scale 1.0 \
--gamma 0.75

# #--output_path /media/allenyljiang/564AFA804AFA5BE51/Codes/cross-image-attention/outputs_debug/ \