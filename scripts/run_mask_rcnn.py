from detectron2.utils.logger import setup_logger
setup_logger()
import argparse, os
from path import Path
import numpy as np
import cv2
import imageio

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
import matplotlib.image as mpimg
from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore')

def run_mask_rcnn(images, output_dir, class_name, _sum=False):

    cfg = get_cfg()
    # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    predictor = DefaultPredictor(cfg)
    number_of_frames = len(images)

    for i in tqdm(range(0, number_of_frames)):
        # try:
        im = np.array(imageio.imread(images[i]))
        #im = cv2.resize(im, (1920, 1080), interpolation=cv2.INTER_LINEAR)
        outputs = predictor(im)
        output_mask_path = output_dir/'{}_mask.png'.format(os.path.split(images[i])[-1].split('.')[0])
        
        if class_name == 'anything':
            if _sum:
                mask = outputs["instances"].pred_masks.sum(0).cpu().numpy()
                for j in range(len(outputs['instances'])):
                    name = predictor.metadata.thing_classes[(outputs['instances'][j].pred_classes.cpu()).long()]
                    print(name)
                    mask += outputs['instances'].pred_masks[j].cpu().numpy()
                imageio.imwrite(output_mask_path, np.uint8(mask * 255.))
            else:
                try:
                    mask = outputs["instances"].pred_masks[0].cpu().numpy()
                    cv2.imwrite(output_mask_path, np.uint8(mask * 255.))
                except:
                    cv2.imwrite(output_mask_path, np.zeros((im.shape[0], im.shape[1])))
        else:
            found_anything = False
            for j in range(len(outputs['instances'])):
                if predictor.metadata.thing_classes[(outputs['instances'][j].pred_classes.cpu()).long()]==class_name:
                    # found the required class, save the mask
                    mask = outputs["instances"].pred_masks[j].cpu().numpy()
                    cv2.imwrite(output_mask_path, np.uint8(mask * 255.))
                    found_anything = True
                    break
                else:
                    # found unneeded class
                    print("Frame %d: Did not find %s, found %s"%(i,class_name,predictor.metadata.thing_classes[(outputs['instances'][j].pred_classes.cpu()).long()]))
            if not found_anything:
                cv2.imwrite(output_mask_path, np.zeros((im.shape[0], im.shape[1])))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_images', type=str)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--class_name', type=str, default='anything')
    parser.add_argument('--sum', action='store_true')
    args = parser.parse_args()
    
    input = Path(args.input_images)
    output = args.output_dir
    if os.path.isdir(input):
        images = sorted(list(input.glob('*.png')) + list(input.glob('*.jpg')))
        if output is None:
            output = input
    else:
        images = [input]
        if output is None:
            output = os.path.split(input)[0]
    
    output = Path(output)
    output.makedirs_p()
    run_mask_rcnn(images, output, args.class_name, args.sum)
