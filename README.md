# [WACV 2026 Oral] VideoSketcher: A Training-Free Approach for Coherent Video Sketch Transfer

[[Paper](https://openaccess.thecvf.com/content/WACV2026/papers/Li_VideoSketcher_A_Training-Free_Approach_for_Coherent_Video_Sketch_Transfer_WACV_2026_paper.pdf)] | [[Video]](https://www.youtube.com/watch?v=WwPmkvbpl5k)

VideoSketcher, a training-free framework for stylecontrollable
sketch video generation that preserves frame
structure while applying specified sketch aesthetics. Leveraging
text-to-image diffusion models.

<img src='assets/figure1.png' width="800">

## Pipeline

Our proposed method's overall pipeline:

<img src='assets/figure2.png' width="800">

## Setup

### Create a Conda Environment

```bash
conda create -n videosketcher python=3.8.5
conda activate videosketcher
```

### Install Dependencies

```bash
# Install all dependencies via pip
pip install -r environment/requirements.txt -i http://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com

# Download NLTK data
python3 -c "import nltk; nltk.download('punkt'); nltk.download('averaged_perceptron_tagger')"
```

### Download Pre-trained Weights

#### StableDiffusion 
Download the StableDiffusion weights from [stable-diffusion-v1-5 at Hugging Face](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5) or your specific SD version.

#### CSD Score (for Evaluation)
Download the CSD Score model for style similarity evaluation from [CSD_Score](https://github.com/haofanwang/CSD_Score). Download the pre-trained weights and place them under `./pretrained_models/CSD_Score`:

```bash
mkdir -p ./pretrained_models/CSD_Score
gdown --id 1FX0xs8p-C7Ob-h5Y4cUhTeOepHzXv_46 -O ./pretrained_models/CSD_Score/
```

#### ArtFID (for Evaluation)
Download the ArtFID model for artistic style evaluation. Place the weights under `./pretrained_models/art_fid/`:

```bash
mkdir -p ./pretrained_models/art_fid
wget https://huggingface.co/matthias-wright/art_inception/resolve/main/art_inception.pth -O ./pretrained_models/art_fid/art_inception.pth
```

## Datasets

### Video Datasets
We evaluate our method on the following video datasets:

- **[DAVIS 2017](https://davischallenge.org/)**: Densely Annotated VIdeo Segmentation dataset, used for video object segmentation and style transfer evaluation.
- **[LOVEU-TGVE 2023](https://github.com/showlab/loveu-tgve-2023)**: Long-form Video Editing and Understanding - Track 2: Text-Guided Video Editing, used for text-guided video editing evaluation.

### Style Datasets
- **[4SKST](https://github.com/Chanuku/4skst)**: 4-Sketch Styles dataset, containing diverse sketch styles used as style references for video sketch transfer.
- **[Sketch Style Dataset](https://drive.google.com/file/d/1lB3Hw6tMDA2SYaCy9X7TqTZAvYkRescb/view?usp=drive_link)**: A collection of sketch style images curated from [Pinterest](https://www.pinterest.com/), used as additional style references in this work.

## Usage

To run the video appearance transfer, execute the main script and pass the path to your configuration file. 

Example running with a dog configuration inversion and reconstruction:
```bash
python3 video_appearance_transfer_model_recon.py --config configs/dog.yaml
```

Example running with a libby configuration video style transfer:
set input_path and app_image_path in the config:
```bash
python3 video_appearance_transfer_model.py --config configs/libby.yaml
```

### Configuration Details

Before running, you need to modify the parameters inside your `.yaml` configuration file. Key parameters include:

* `input_path`: The directory path to your original input video.
* `sd_version`: The specific Stable Diffusion version you are using.
* `model_key`: The local folder path where your downloaded Stable Diffusion model files are saved.
* `app_image_path`: The path to the style reference image.
* `app_image_save_path`: The directory path to store the results after style inversion.

## Acknowledgements

This project is highly built on [Cross-Image-Attention](https://github.com/garibida/Cross-Image-Attention), [StyleID](https://github.com/jiwoogit/StyleID.git), and [Edit-Friendly DDPM Inversion](https://github.com/inbarhub/DDPM_inversion).

## Citation

If you use the code and models please cite:

```bibtex
@inproceedings{li2026videosketcher,
  title={VideoSketcher: A Training-Free Approach for Coherent Video Sketch Transfer},
  author={Li, Huining and Liu, Bangzhen and Yang, Rui and Zhou, Yang and Xu, Chenshu and Pang, Xufang and He, Shengfeng},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
  year={2026}
}
```