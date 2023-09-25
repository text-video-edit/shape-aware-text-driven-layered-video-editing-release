import sys, os, argparse, yaml
import numpy as np
from PIL import Image
from path import Path
import imageio
import cv2 as cv
sys.path[0] = os.path.join(sys.path[0], '..')
from utils import *

MAX_SIZE = 512

def crop_foreground(image, mask, mask_area='dilate', dilate_iter=20, padding=50):
    # get mask for inpainting
    if mask_area == 'dilate':
        mask = cv.dilate(mask, np.ones((5, 5), np.uint8), iterations=dilate_iter)
    elif mask_area == 'rect':
        t, b, l, r = find_mask_border(mask, pad=int(dilate_iter*2))
        mask[t:b, l:r] = 255
    else:
        raise NotImplementedError

    # crop foreground area 
    t, b, l, r = find_mask_border(mask, pad=padding)
    H, W = mask.shape
    w, h = r - l, b - t

    ## keep aspect ratio
    aspect_ratio = 1 # H / W
    if h / w > aspect_ratio:
        w_ = h * (1 / aspect_ratio)
        l = max(int(l - (w_ - w) * 0.5), 0)
        r = min(int(r + (w_ - w) * 0.5), W - 1)
    else:
        h_ = w * aspect_ratio
        t = max(int(t - (h_ - h) * 0.5), 0)
        b = min(int(b + (h_ - h) * 0.5), H - 1)

    # resize to the multiples of 64 for diffusion models
    w, h = r - l, b - t
    factor = MAX_SIZE / max(w, h)
    w = int(np.round(factor * w))
    h = int(np.round(factor * h))
    w, h = map(lambda x: int(x - x % 64), (w, h))
    w, h = 512, 512
    # crop and resize
    image_crop = cv.resize(image[t:b, l:r], (w, h), interpolation=cv.INTER_LINEAR)
    mask_crop = cv.resize(mask[t:b, l:r], (w, h), interpolation=cv.INTER_LINEAR)

    return image_crop, mask_crop, [t / (H - 1), b / (H - 1), l / (W - 1), r / (W - 1)]

def resize(img, size=None, max_size=1024):
    h, w = img.shape[:2]
    if size is None:
        ratio = max_size / max(h, w)
        img_resized = cv.resize(img, None, fx=ratio, fy=ratio, interpolation=cv.INTER_LINEAR)
    else:
        img_resized = cv.resize(img, size, interpolation=cv.INTER_LINEAR)
    return img_resized


def edit_keyframe(path_img, path_mask, output_dir, args):
    path_input_image = output_dir/'keyframe_input.png'
    path_edited_image = output_dir/'keyframe_edited.png'
    path_input_crop = output_dir/'keyframe_input_crop.png'
    path_edited_crop = output_dir/'keyframe_edited_crop.png'
    path_input_mask = output_dir/'keyframe_input_mask.png'
    path_edited_mask = output_dir/'keyframe_edited_mask.png'
    path_input_crop_mask = output_dir/'keyframe_input_crop_mask.png'
    path_edited_crop_mask = output_dir/'keyframe_edited_crop_mask.png'

    input_image = imageio.imread(path_img)
    input_mask = imageio.imread(path_mask)
    imageio.imwrite(path_input_image, input_image)
    imageio.imwrite(path_input_mask, input_mask)
    H, W = input_mask.shape
    
    input_crop, input_mask_crop, crop_area = crop_foreground(input_image, input_mask)
    imageio.imwrite(path_input_crop, input_crop)
    imageio.imwrite(path_input_crop_mask, input_mask_crop)

    sd_model = load_sd_model(args['sd_mode']).to('cuda')
    
    if 'inpaint' in args['sd_mode']:
        edited_crop = sd_model(
            prompt=[args['prompt']],
            image=Image.fromarray(input_crop),
            mask_image=Image.fromarray(input_mask_crop)
        ).images[0]
        edited_crop = np.array(edited_crop)
    else:
        raise NotImplementedError
    
    output_dir.makedirs_p()
    imageio.imwrite(path_edited_crop, edited_crop)
    edited_crop = resize(get_superresolution(path_edited_crop)) # make it sharper
    imageio.imwrite(path_edited_crop, edited_crop)
    edited_crop_mask = get_mask(path_edited_crop, edit_dir)

    edited_image = input_image.copy()
    args['crop_area'] = np.array(crop_area).clip(0, 1).tolist()
    t, b, l, r = args['crop_area']
    t = int(t * (H - 1)); b = int(b * (H - 1)); l = int(l * (W - 1)); r = int(r * (W - 1))
    edited_image[t:b, l:r] = resize(edited_crop, size=(r - l, b - t))
    imageio.imwrite(path_edited_image, edited_image)

    edited_mask = input_mask.copy()
    edited_mask[t:b, l:r] = resize(edited_crop_mask, size=(r - l, b - t))
    imageio.imwrite(path_edited_mask, edited_mask)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type=str)
    parser.add_argument('text_prompt', type=str)
    parser.add_argument('--kf', type=str, default='mid')
    parser.add_argument('--strength', type=float, default=0.3)
    parser.add_argument('--guidance_scale', type=float, default=4)
    parser.add_argument('--sd_mode', type=str, choices=['inpaint'], default='inpaint')
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    nla_dir = data_dir/'nla_outputs'
    
    # create a directory to save video editing result
    edit_dir = data_dir/'edit_{}'.format(args.text_prompt.split(',')[0].replace(' ', '_'))
    edit_dir.makedirs_p()

    # select keyframe
    frames = sorted(list((data_dir/'images').glob('*.png')) + list((data_dir/'images').glob('*.jpg')))[:70]
    masks = sorted(list((data_dir/'masks').glob('*.png')) + list((data_dir/'masks').glob('*.jpg')))[:70]
    N = len(frames)

    kf_path = None
    if args.kf == 'mid':
        idx = N // 2
    else:
        idx = int(args.kf)

    kf_path = frames[idx]
    kf_mask_path = masks[idx]
     
    edit_meta = {
        'prompt': args.text_prompt,
        'keyframe_index': idx,
        'strength': args.strength,
        'guidance_scale': args.guidance_scale,
        'sd_mode': args.sd_mode,
    }

    edit_keyframe(kf_path, kf_mask_path, edit_dir, edit_meta)

    f = open(edit_dir/'edit_meta.yaml', 'w')
    yaml.dump(edit_meta, f)
