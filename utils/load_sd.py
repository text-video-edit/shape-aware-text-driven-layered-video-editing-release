from diffusers import StableDiffusionImg2ImgPipeline, StableDiffusionInpaintPipeline
from path import Path
import os
import torch

version_downloads = {
        '1.4': 'CompVis/stable-diffusion-v1-4',
        '1.5': 'runwayml/stable-diffusion-v1-5',
        '2.0': 'stabilityai/stable-diffusion-2',
        'inpaint': 'stabilityai/stable-diffusion-2-inpainting',
}


def load_sd_model(version, save_dir='pretrained-models'):
    local_path = Path(save_dir)/('sd-' + version)
    
    if not os.path.isdir(local_path):
        # download pretrained
        if version in list(version_downloads.keys()):
            download_path = version_downloads[version]
            print('=> Download stable-diffusion from {}'.format(download_path))
        else:
            raise NotImplementedError('Indicated version not available')
        
        if 'inpaint' in version:
            model = StableDiffusionInpaintPipeline.from_pretrained(download_path, revision='fp16', torch_dtype=torch.float16)
        else:
            model = StableDiffusionImg2ImgPipeline.from_pretrained(download_path)

        model.save_pretrained(local_path)
    else:
        # load pretrained from local
        print('=> Load stable-diffusion from local {}'.format(local_path))
        if 'inpaint' in version:
            model = StableDiffusionInpaintPipeline.from_pretrained(local_path, revision='fp16', torch_dtype=torch.float16)
        else:
            model = StableDiffusionImg2ImgPipeline.from_pretrained(local_path)
            

    return model

