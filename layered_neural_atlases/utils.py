import matplotlib as mpl

mpl.use('Agg')
import skimage.metrics
import io
from scipy.interpolate import griddata
import matplotlib.image as mpimg
import torch
import colorsys

import matplotlib.pyplot as plt
import skimage.measure
import os
from PIL import Image
import cv2
import numpy as np
import imageio


# taken from https://gist.github.com/peteflorence/a1da2c759ca1ac2b74af9a83f69ce20e
# sample coordinates x,y from image im.
def bilinear_interpolate_numpy(im, x, y):
    x0 = np.floor(x).astype(int)
    x1 = x0 + 1
    y0 = np.floor(y).astype(int)
    y1 = y0 + 1

    x0 = np.clip(x0, 0, im.shape[1] - 1)
    x1 = np.clip(x1, 0, im.shape[1] - 1)
    y0 = np.clip(y0, 0, im.shape[0] - 1)
    y1 = np.clip(y1, 0, im.shape[0] - 1)

    Ia = im[y0, x0]
    Ib = im[y1, x0]
    Ic = im[y0, x1]
    Id = im[y1, x1]

    wa = (x1 - x) * (y1 - y)
    wb = (x1 - x) * (y - y0)
    wc = (x - x0) * (y1 - y)
    wd = (x - x0) * (y - y0)

    return (Ia.T * wa).T + (Ib.T * wb).T + (Ic.T * wc).T + (Id.T * wd).T

# Given target (non integer) uv coordinates with corresponding alpha values create an
# 1000x1000 uv image of alpha values
def interpolate_alpha(m1_alpha_v, m1_alpha_u, m1_alpha_alpha):
    xv, yv = np.meshgrid(np.linspace(0, 999, 1000), np.linspace(0, 999, 1000))

    m1_alpha_u = np.concatenate(m1_alpha_u)
    m1_alpha_v = np.concatenate(m1_alpha_v)
    m1_alpha_alpha = np.concatenate(m1_alpha_alpha)

    masks1 = griddata((m1_alpha_u, m1_alpha_v), m1_alpha_alpha,
                      (xv, yv), method='linear')
    return masks1

# Given uv points in the range (-1,1) and an image (with a given "resolution") that represents a crop (defined by "minx", "maxx", "miny", "maxy")
# Change uv points to pixel coordinates, and sample points from the image
def get_colors(resolution, minx, maxx, miny, maxy, pointx, pointy, image):
    pixel_size = resolution / (maxx - minx)
    # Change uv to pixel coordinates of the discretized image
    pointx2 = ((pointx - minx) * pixel_size).numpy()
    pointy2 = ((pointy - miny) * pixel_size).numpy()
    # Bilinear interpolate pixel colors from the image
    pixels = bilinear_interpolate_numpy(image.numpy(), pointx2, pointy2)
    
    # Relevant pixel locations should be positive:
    pos_logicaly = np.logical_and(np.ceil(pointy2) >= 0, np.floor(pointy2) >= 0)
    pos_logicalx = np.logical_and(np.ceil(pointx2) >= 0, np.floor(pointx2) >= 0)
    pos_logical = np.logical_and(pos_logicaly, pos_logicalx)

    # Relevant pixel locations should be inside the image borders:
    mx_logicaly = np.logical_and(np.ceil(pointy2) < resolution, np.floor(pointy2) < resolution)
    mx_logicaxlx = np.logical_and(np.ceil(pointx2) < resolution, np.floor(pointx2) < resolution)
    mx_logical = np.logical_and(mx_logicaly, mx_logicaxlx)

    # Relevant should satisfy both conditions
    relevant = np.logical_and(pos_logical, mx_logical)

    return pixels[relevant], pointx2[relevant], pointy2[relevant], relevant


# Sample discrete atlas image from a neural atlas
def get_high_res_texture(resolution, minx, maxx, miny, maxy, model_F_atlas,device
                         ):
    indsx = torch.linspace(minx, maxx, resolution)
    indsy = torch.linspace(miny, maxy, resolution)
    reconstruction_texture2 = torch.zeros((resolution, resolution, 3))
    counter = 0
    with torch.no_grad():

        # reconsruct image row by row
        for i in indsy:
            reconstruction_texture2[counter, :, :] = model_F_atlas(
                torch.cat((indsx.unsqueeze(1), i * torch.ones_like(indsx.unsqueeze(1))),
                          dim=1).to(device)).detach().cpu()
            counter = counter + 1
        # move colors to RGB color domain (0,1)
        reconstruction_texture2 = 0.5 * (reconstruction_texture2 + 1)

        reconsturction_texture2_orig = reconstruction_texture2.clone()

        # Add text pattern to the texture, in order to visualize the mapping functions.
        for ii in range(40, 500, 80):
            cur_color = colorsys.hsv_to_rgb((ii - 40) / 500, 1.0, 1.0)
            cv2.putText(reconstruction_texture2.numpy(),
                        'abcdefghijlmnopqrstuvwxyz1234567890!@#$%^&*()-+=>', (10, ii),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, cur_color, 2, cv2.LINE_AA)
            cv2.putText(reconstruction_texture2.numpy(),
                        'ABCDEFGHIJKLMNOPQRSTUVWXYZ?~;:<./\|][{},', (10, ii + 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, cur_color, 2, cv2.LINE_AA)

        for ii in range(40, 500, 80):
            cur_color = colorsys.hsv_to_rgb((ii - 40) / 500, 1.0, 1.0)
            cv2.putText(reconstruction_texture2.numpy(),
                        'abcdefghijlmnopqrstuvwxyz1234567890!@#$%^&*()-+=>', (10, ii + 500),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, cur_color, 2, cv2.LINE_AA)
            cv2.putText(reconstruction_texture2.numpy(),
                        'ABCDEFGHIJKLMNOPQRSTUVWXYZ?~;:<./\|][{},',
                        (10, ii + 40 + 500), cv2.FONT_HERSHEY_SIMPLEX, 1.1, cur_color, 2,
                        cv2.LINE_AA)

        return reconstruction_texture2, reconsturction_texture2_orig


def get_img_from_fig(fig, dpi=180):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img

# Given a mapping model "model_F_mapping", maskRCNN masks "mask_frames" and alpha network "F_alpha", find
# a range in the uv domain, that maskRCNN points are mapped to by model_F_mapping.
def get_mapping_area(model_F_mapping, F_alpha, mask_frames, resx, number_of_frames, uv_shift, device,
                     invert_alpha=False,
                     alpha_thresh=-0.5):
    # consider only pixels that their masks are 1
    relis_i, relis_j, relis_f = torch.where(mask_frames)

    # split all i,j,f coordinates to batches of size 100k
    relisa = np.array_split(relis_i.numpy(), np.ceil(relis_i.shape[0] / 100000))
    reljsa = np.array_split(relis_j.numpy(), np.ceil(relis_i.shape[0] / 100000))
    relfsa = np.array_split(relis_f.numpy(), np.ceil(relis_i.shape[0] / 100000))

    minx = 1
    miny = 1
    maxx = -1
    maxy = -1
    with torch.no_grad():
        for i in range(len(relisa)):
            relis = torch.from_numpy(relisa[i]).unsqueeze(1) / (resx / 2) - 1
            reljs = torch.from_numpy(reljsa[i]).unsqueeze(1) / (resx / 2) - 1
            relfs = torch.from_numpy(relfsa[i]).unsqueeze(1) / (number_of_frames / 2) - 1

            uv = model_F_mapping(torch.cat((reljs, relis, relfs),
                                           dim=1).to(device)).cpu()
            alpha = F_alpha(torch.cat((reljs, relis, relfs),
                                      dim=1).to(device)).cpu().squeeze()
            if invert_alpha:
                alpha = -alpha
            if torch.any(alpha > alpha_thresh):
                uv = uv * 0.5 + uv_shift
                curminx = torch.min(uv[alpha > alpha_thresh, 0])
                curminy = torch.min(uv[alpha > alpha_thresh, 1])
                curmaxx = torch.max(uv[alpha > alpha_thresh, 0])
                curmaxy = torch.max(uv[alpha > alpha_thresh, 1])
                minx = torch.min(torch.tensor([curminx, minx]))
                miny = torch.min(torch.tensor([curminy, miny]))

                maxx = torch.max(torch.tensor([curmaxx, maxx]))
                maxy = torch.max(torch.tensor([curmaxy, maxy]))

    maxx = np.minimum(maxx, 1)
    maxy = np.minimum(maxy, 1)

    minx = np.maximum(minx, -1)
    miny = np.maximum(miny, -1)

    edge_size = torch.max(torch.tensor([maxx - minx, maxy - miny]))
    return maxx, minx, maxy, miny, edge_size

# for visualizing uv images their values are mapped to the range (0,1) by using (edge_size, minx, miny)
# which represent the information about the mapping range. The idea is to stretch this range to (0,1).
def normalize_uv_images(uv_frames_reconstruction, values_shift, edge_size, minx, miny):
    uv_frames_reconstruction[:, :, 0, :] = ((uv_frames_reconstruction[:, :, 0, :] * 0.5 + values_shift) - np.float64(
        minx)) / edge_size
    uv_frames_reconstruction[:, :, 1, :] = ((uv_frames_reconstruction[:, :, 1, :] * 0.5 + values_shift) - np.float64(
        miny)) / edge_size
    uv_frames_reconstruction[uv_frames_reconstruction > 1] = 1
    uv_frames_reconstruction[uv_frames_reconstruction < 0] = 0
    return uv_frames_reconstruction
