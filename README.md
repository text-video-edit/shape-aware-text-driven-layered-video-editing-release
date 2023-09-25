# Shape-aware Text-driven Layered Video Editing [CVPR 2023]

[Yao-Chih Lee](https://yaochih.github.io/),
Ji-Ze G. Jang,
[Yi-Ting Chen](https://jamie725.github.io/website/),
[Elizabeth Qiu](https://elizabethqiu.com/),
[Jia-Bin Huang](https://jbhuang0604.github.io/)

[[Webpage](https://text-video-edit.github.io/)] 
[[Paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Lee_Shape-Aware_Text-Driven_Layered_Video_Editing_CVPR_2023_paper.pdf)]

### Environment
- Tested on Pytorch 1.12.1 and CUDA 11.3


```
git clone --recursive 
pip install -r requirements.txt
```

- Download model weights
```
./scripts/download_models.sh
```

### Data structure
For an input video, the required data and structure are listed below. The NLA's checkpoint and configuration files are obtained by [Layered Neural Atlases](https://github.com/ykasten/layered-neural-atlases).
```
DATA_DIR/
├── images/
│   └── *.png or *.jpg
├── masks
│   └── *.png 
└── pretrained_nla_models
    ├── checkpoint
    └── config.json
```

Each edit case will be saved in `EDIT_DIR`, which is put under the `DATA_DIR`. We provided some examples in `data` directory. 

For instance, `DATA_DIR=data/car-turn` and  `EDIT_DIR=data/car-turn/edit_sports_car`.

### Running

- Generate NLA outputs from NLA pretrained model
  ```
  python scripts/generate_nla_outputs.py [DATA_DIR]
  ```
- Edit Foreground 
  ```
	python scripts/edit_foreground [DATA_DIR] [TEXT_PROMPT]
  ```
  Please put your HuggingFace token file named `TOKEN` in root directory. 
  It will create the `EDIT_DIR` under `DATA_DIR` and all the keyframe's data will be saved in `EDIT_DIR`. Note that we may manually refine the mask of the edited keyframe sometime since MaskRCNN may fail to find precise masks for diffusion-generated images.

- Semantic correspondence
  To achieve the best editing results, we use the warping tools in Photoshop to obtain the semantic correspondence between `EDIT_DIR/keyframe_input_crop.png` and `EDIT_DIR/keyframe_edited_crop.png`, and saved as `EDIT_DIR/semantic_correspondence_crop.npy`. The correspondence format is similar to optical flow, ranging from [-1, 1].

- Optimization
  ```
  python main.py [EDIT_DIR]
  ```
  The training results will be saved in `EDIT_DIR/workspace`

### Acknowledgements
We thank the authors for releasing 
[Layered Neural Atlases](https://github.com/ykasten/layered-neural-atlases), 
[Text2LIVE](https://github.com/omerbt/Text2LIVE), 
[Stable-DreamFusion](https://github.com/ashawkey/stable-dreamfusion), and 
[Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN).
