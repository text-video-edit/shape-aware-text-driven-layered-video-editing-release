import sys, os, argparse, json, time
sys.path[0] = os.path.join(sys.path[0], '..')
#from nla_implicit_neural_networks import IMLP
import matplotlib.image as mpimg
import time
from scipy.interpolate import griddata
import torch
import numpy as np
import imageio
import cv2
from PIL import Image
from tqdm import tqdm
import imageio
from path import Path
from layered_neural_atlases.utils import *
from layered_neural_atlases.implicit_neural_networks import IMLP

from utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

uv2_meta = {}

def get_nla_outputs(resx, resy, number_of_frames, models, texture_resolution=2000):

    larger_dim = np.maximum(resx, resy)

    # get relevant working crops from the atlases for atlas discretization
    minx = 0
    miny = 0
    edge_size = 1
   
    alpha_maps = torch.zeros((number_of_frames, resy, resx))
    uv_maps1 = torch.zeros((number_of_frames, resy, resx, 2))
    uv_maps2 = torch.zeros((number_of_frames, resy, resx, 2))

    maxx2, minx2, maxy2, miny2, edge_size2 = get_mapping_area(models['F_mapping2'], 
                                                              models['alpha'], 
                                                              alpha_maps.permute(1, 2, 0) > -1, 
                                                              larger_dim,
                                                              number_of_frames,
                                                              torch.tensor([-0.5, -0.5]), # uv shift
                                                              device, 
                                                              invert_alpha=True)

    edited_tex1, texture1 = get_high_res_texture(
        texture_resolution,
        0, 1, 0, 1, 
        models['F_atlas'],
        device)
    
    edited_tex2, texture2 = get_high_res_texture(
        texture_resolution,
        minx2, minx2 + edge_size2, miny2, miny2 + edge_size2,  
        models['F_atlas'], 
        device)

    global uv2_meta 
    uv2_meta = {
        'minx2': minx2,
        'miny2': miny2,
        'edge_size2': edge_size2,
    }

    
    with torch.no_grad():
        for t in tqdm(range(number_of_frames)):
            uv1 = forward_model(models['F_mapping1'], 'uv', resx, resy, t / number_of_frames)
            uv2 = forward_model(models['F_mapping2'], 'uv', resx, resy, t / number_of_frames)
            alpha = forward_model(models['alpha'], 'alpha', resx, resy, t / number_of_frames)

            uv2 = uv2 * 0.5 - 0.5
            uv2[..., 0] = (uv2[..., 0] - minx2) / edge_size2 * 2 - 1
            uv2[..., 1] = (uv2[..., 1] - miny2) / edge_size2 * 2 - 1
            
            alpha = (alpha + 1.) * 0.5 * 0.99 + 0.001

            uv_maps1[[t]] = uv1
            uv_maps2[[t]] = uv2
            alpha_maps[[t]] = alpha

    return texture1, texture2, uv_maps1, uv_maps2, alpha_maps

def forward_model(model, mode, resx, resy, t, crop_area=None):
    grid_y, grid_x = torch.where(torch.ones(resy, resx) > 0)
    mesh_grid = torch.stack((grid_x, grid_y), dim=1).to(device).float()

    if crop_area is not None:
        mesh_grid[:, 0] = mesh_grid[:, 0] / (resx - 1) * 2 - 1
        mesh_grid[:, 1] = mesh_grid[:, 1] / (resy - 1) * 2 - 1
        mesh_grid = rescale_from_crop(mesh_grid, crop_area)
        mesh_grid[:, 0] = (mesh_grid[:, 0] + 1) * 0.5 * (resx - 1)
        mesh_grid[:, 1] = (mesh_grid[:, 1] + 1) * 0.5 * (resy - 1)

    mesh_grid = mesh_grid / np.maximum(resx, resy) * 2. - 1.

    coords = torch.cat((mesh_grid, (t * 2. - 1) * torch.ones(resy * resx, 1).to(device)), dim=1)
            
    pixel_batch_size = 100000
    begin_i, end_i = 0, 0
    if mode == 'uv':
        tensor = torch.ones(resy * resx, 2).float().to(device) * -10
    else:
        tensor = torch.zeros(resy * resx, 1).to(device)
    
    while end_i < coords.shape[0]:
        end_i = min(begin_i + pixel_batch_size, coords.shape[0])
        
        tensor[begin_i:end_i] = model(coords[begin_i:end_i])
        
        begin_i = end_i
    
    if mode == 'uv':
        tensor = tensor.view(1, resy, resx, 2).cpu()
    else:
        tensor = tensor.view(1, resy, resx).cpu()
    return tensor

def resemble(texture1, texture2, maps, output_video_path):
    '''
    Input:
        texture1: [3, H, W]
        texture2: [3, H, W]
    '''
    uv1 = maps['uv1'].to(device)
    uv2 = maps['uv2'].to(device)
    alpha = maps['alpha'].to(device)

    number_frames, resy, resx = alpha.shape
    
    texture1 = texture1.unsqueeze(0).to(device)
    texture2 = texture2.unsqueeze(0).to(device)

    video_writer = imageio.get_writer(output_video_path, fps=10)
    with torch.no_grad():
        for t in tqdm(range(number_frames)):
            img1 = F.grid_sample(texture1, uv1[t:t+1], mode='bilinear').squeeze(0)
            img2 = F.grid_sample(texture2, uv2[t:t+1], mode='bilinear').squeeze(0)
            
            img = img1 * alpha[t:t+1, :, :] + img2 * (1 - alpha[t:t+1, :, :])
            
            video_writer.append_data(np.uint8(img.permute(1, 2, 0).cpu().numpy() * 255.))

    video_writer.close()

def load_model(data_dir):
    
    model_checkpoint_path = data_dir/'pretrained_nla_models/checkpoint'
    model_config_path = data_dir/'pretrained_nla_models/config.json'
    
    with open(model_config_path) as f:
        config = json.load(f)

    maximum_number_of_frames = config["maximum_number_of_frames"]
    resx = np.int64(config["resx"])
    resy = np.int64(config["resy"])

    positional_encoding_num_alpha = config["positional_encoding_num_alpha"]

    number_of_channels_atlas = config["number_of_channels_atlas"]
    number_of_layers_atlas = config["number_of_layers_atlas"]

    number_of_channels_alpha = config["number_of_channels_alpha"]
    number_of_layers_alpha = config["number_of_layers_alpha"]

    use_positional_encoding_mapping1 = config["use_positional_encoding_mapping1"]
    number_of_positional_encoding_mapping1 = config["number_of_positional_encoding_mapping1"]
    number_of_layers_mapping1 = config["number_of_layers_mapping1"]
    number_of_channels_mapping1 = config["number_of_channels_mapping1"]

    use_positional_encoding_mapping2 = config["use_positional_encoding_mapping2"]
    number_of_positional_encoding_mapping2 = config["number_of_positional_encoding_mapping2"]
    number_of_layers_mapping2 = config["number_of_layers_mapping2"]
    number_of_channels_mapping2 = config["number_of_channels_mapping2"]


    input_files = list((data_dir/'images').glob('*.jpg')) + list((data_dir/'images').glob('*.png'))
    number_of_frames = np.minimum(maximum_number_of_frames, len(input_files))

    # Define MLPs
    model_F_mapping1 = IMLP(
        input_dim=3,
        output_dim=2,
        hidden_dim=number_of_channels_mapping1,
        use_positional=use_positional_encoding_mapping1,
        positional_dim=number_of_positional_encoding_mapping1,
        num_layers=number_of_layers_mapping1,
        skip_layers=[]).to(device)
    model_F_mapping2 = IMLP(
        input_dim=3,
        output_dim=2,
        hidden_dim=number_of_channels_mapping2,
        use_positional=use_positional_encoding_mapping2,
        positional_dim=number_of_positional_encoding_mapping2,
        num_layers=number_of_layers_mapping2,
        skip_layers=[]).to(device)

    model_F_atlas = IMLP(
        input_dim=2,
        output_dim=3,
        hidden_dim=number_of_channels_atlas,
        use_positional=True,
        positional_dim=10,
        num_layers=number_of_layers_atlas,
        skip_layers=[4, 7]).to(device)

    model_alpha = IMLP(
        input_dim=3,
        output_dim=1,
        hidden_dim=number_of_channels_alpha,
        use_positional=True,
        positional_dim=positional_encoding_num_alpha,
        num_layers=number_of_layers_alpha,
        skip_layers=[]).to(device)

    checkpoint = torch.load(model_checkpoint_path)

    model_F_atlas.load_state_dict(checkpoint["F_atlas_state_dict"])
    model_F_atlas.eval()
    model_F_atlas.to(device)

    model_F_mapping1.load_state_dict(checkpoint["model_F_mapping1_state_dict"])
    model_F_mapping1.eval()
    model_F_mapping1.to(device)

    model_F_mapping2.load_state_dict(checkpoint["model_F_mapping2_state_dict"])
    model_F_mapping2.eval()
    model_F_mapping2.to(device)

    model_alpha.load_state_dict(checkpoint["model_F_alpha_state_dict"])
    model_alpha.eval()
    model_alpha.to(device)

    models = {
        'F_atlas': model_F_atlas,
        'F_mapping1': model_F_mapping1,
        'F_mapping2': model_F_mapping2,
        'alpha': model_alpha
    }
    return resx, resy, number_of_frames, models

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type=str)
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    
    resx, resy, number_of_frames, models = load_model(data_dir)

    output_dir = data_dir/'nla_outputs'
    output_dir.makedirs_p()
     
    texture1, texture2, uv1, uv2, alpha = get_nla_outputs(resx, resy, number_of_frames, models)
    
    masks = read_images(sorted(list((data_dir/'masks').glob('*.png'))), (resy, resx))

    texture1, texture1_crop_area = crop_texture(texture1.permute(2, 0, 1)[None, :, :, :], uv1, masks)
    texture1 = texture1[0].permute(1, 2, 0)
    uv1 = scale_to_crop(uv1, texture1_crop_area)
    
    texture2, texture2_crop_area = crop_texture(texture2.permute(2, 0, 1)[None, :, :, :], uv2)
    texture2 = texture2[0].permute(1, 2, 0)
    uv2 = scale_to_crop(uv2, texture2_crop_area)
    
    crop_areas1 = []
    uv1_crop = torch.zeros((number_of_frames, resy, resx, 2))
    uv2_crop = torch.zeros((number_of_frames, resy, resx, 2))
    alpha_crop = torch.zeros((number_of_frames, resy, resx))
    with torch.no_grad():
        for t in range(uv1.shape[0]):
            crop_area = get_crop_area(masks[t], (resy, resx), pad=25)
            crop_areas1.append(crop_area)
            uv1_crop_temp = forward_model(models['F_mapping1'], 'uv', resx, resy, t / number_of_frames, crop_area)
            uv2_crop_temp = forward_model(models['F_mapping2'], 'uv', resx, resy, t / number_of_frames, crop_area)
            alpha_crop_temp = forward_model(models['alpha'], 'alpha', resx, resy, t / number_of_frames, crop_area)

            alpha_crop_temp = (alpha_crop_temp + 1.) * 0.5 * 0.99 + 0.001
            
            minx2, miny2, edge_size2 = uv2_meta['minx2'], uv2_meta['miny2'], uv2_meta['edge_size2']
            uv2_crop_temp = uv2_crop_temp * 0.5 - 0.5
            uv2_crop_temp[..., 0] = (uv2_crop_temp[..., 0] - minx2) / edge_size2 * 2 - 1
            uv2_crop_temp[..., 1] = (uv2_crop_temp[..., 1] - miny2) / edge_size2 * 2 - 1
            
            uv1_crop[[t]] = uv1_crop_temp
            uv2_crop[[t]] = uv2_crop_temp
            alpha_crop[[t]] = alpha_crop_temp

    uv1_crop = scale_to_crop(uv1_crop, texture1_crop_area)
    uv2_crop = scale_to_crop(uv2_crop, texture2_crop_area)

    print('texture1: ', texture1_crop_area)
    print('texture2: ', texture2_crop_area)

    imageio.imwrite(output_dir/'texture1.png', np.uint8(texture1.numpy() * 255))
    imageio.imwrite(output_dir/'texture2.png', np.uint8(texture2.numpy() * 255))
    maps = {
        'uv1': uv1,
        'uv2': uv2,
        'alpha': alpha,
        'uv1_crop': uv1_crop,
        'uv2_crop': uv2_crop,
        'alpha_crop': alpha_crop,
        'crop_areas1': torch.FloatTensor(crop_areas1),
    }
    torch.save(maps, output_dir/'maps')
    resemble(texture1.permute(2, 0, 1), texture2.permute(2, 0, 1), maps, output_dir/'resemble.mp4') 

