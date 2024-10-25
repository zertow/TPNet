#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.

import os
import sys
sys.path.append('../')
import argparse
import numpy as np
from tqdm import tqdm
import re
import datetime
import PIL
import PIL.Image as Image
import torch
import torch.nn.functional as F
from torchvision import transforms
from pycocotools import mask
from pycocotools.mask import encode
import pycocotools.mask as mask_util
from scipy import ndimage
from scipy.linalg import eigh
import json
import time

import selective_search
import skimage.io
from transformers import CLIPProcessor,CLIPModel
import cv2
_CONTOUR_INDEX = 1 if cv2.__version__.split('.')[0] == '3' else 0

import matplotlib.pyplot as plt
import dino
import pandas as pd
import torch
from DETReg.datasets.coco import make_coco_transforms
from PIL import Image
from DETReg.main import get_args_parser
from DETReg.models import build_model
from argparse import Namespace


from pytorch_grad_cam.utils.image import scale_cam_image
from DETReg.util.box_ops import box_cxcywh_to_xyxy
from DETReg.util.plot_utils import plot_results
from matplotlib import pyplot as plt
import numpy as np
import os
import glob
from third_party.TokenCut.unsupervised_saliency_detection import utils, metric
from third_party.TokenCut.unsupervised_saliency_detection.object_discovery import detect_box
from crf import densecrf

import clip
from pytorch_grad_cam import GradCAM

import warnings
warnings.filterwarnings("ignore")

# Image transformation applied to all images
ToTensor = transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize(
                                (0.485, 0.456, 0.406),
                                (0.229, 0.224, 0.225)),])

def get_affinity_matrix(feats, tau, eps=1e-5):
    # get affinity matrix via measuring patch-wise cosine similarity
    feats = F.normalize(feats, p=2, dim=0)
    A = (feats.transpose(0,1) @ feats).cpu().numpy()
    # convert the affinity matrix to a binary one.
    A = A > tau
    A = np.where(A.astype(float) == 0, eps, A)
    d_i = np.sum(A, axis=1)
    D = np.diag(d_i)
    return A, D

def second_smallest_eigenvector(A, D):
    # get the second smallest eigenvector from affinity matrix
    #_, eigenvectors = eigh(D-A, D, eigvals=(1,2))
    _, eigenvectors = eigh(D-A, D, subset_by_index=[1,2])
    eigenvec = np.copy(eigenvectors[:, 0])
    second_smallest_vec = eigenvectors[:, 0]
    return eigenvec, second_smallest_vec

def get_salient_areas(second_smallest_vec):
    # get the area corresponding to salient objects.
    avg = np.sum(second_smallest_vec) / len(second_smallest_vec)
    bipartition = second_smallest_vec > avg
    return bipartition

def check_num_fg_corners(bipartition, dims):
    # check number of corners belonging to the foreground
    bipartition_ = bipartition.reshape(dims)
    top_l, top_r, bottom_l, bottom_r = bipartition_[0][0], bipartition_[0][-1], bipartition_[-1][0], bipartition_[-1][-1]
    nc = int(top_l) + int(top_r) + int(bottom_l) + int(bottom_r)
    return nc

def get_masked_affinity_matrix(painting, feats, mask, ps):
    # mask out affinity matrix based on the painting matrix 
    dim, num_patch = feats.size()[0], feats.size()[1]
    painting = painting + mask.unsqueeze(0)
    painting[painting > 0] = 1
    painting[painting <= 0] = 0
    feats = feats.clone().view(dim, ps, ps)
    feats = ((1 - painting) * feats).view(dim, num_patch)
    return feats, painting

def read_txt_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    line_new = []
    for i in lines:
        q = i[:-1]
        line_new.append(q)
    return line_new
def scoremap2bbox(scoremap, threshold, multi_contour_eval=False):
    height, width = scoremap.shape
    scoremap_image = np.expand_dims((scoremap * 255).astype(np.uint8), 2)
    _, thr_gray_heatmap = cv2.threshold(
        src=scoremap_image,
        thresh=int(threshold * np.max(scoremap_image)),
        maxval=255,
        type=cv2.THRESH_BINARY)
    contours = cv2.findContours(
        image=thr_gray_heatmap,
        mode=cv2.RETR_TREE,
        method=cv2.CHAIN_APPROX_SIMPLE)[_CONTOUR_INDEX]

    if len(contours) == 0:
        return np.asarray([[0, 0, 0, 0]]), 1

    if not multi_contour_eval:
        contours = [max(contours, key=cv2.contourArea)]

    estimated_boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        x0, y0, x1, y1 = x, y, x + w, y + h
        x1 = min(x1, width - 1)
        y1 = min(y1, height - 1)
        estimated_boxes.append([x0, y0, x1, y1])

    return np.asarray(estimated_boxes), len(contours)
def ss_box(im):
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()

    ss.addImage(im)
    gs = cv2.ximgproc.segmentation.createGraphSegmentation()
    # gs.setK(150)
    # gs.setSigma(0.8)
    ss.addGraphSegmentation(gs)
    strategy_color = cv2.ximgproc.segmentation.createSelectiveSearchSegmentationStrategyColor()
    strategy_fill = cv2.ximgproc.segmentation.createSelectiveSearchSegmentationStrategyFill()
    strategy_size = cv2.ximgproc.segmentation.createSelectiveSearchSegmentationStrategySize()
    strategy_texture = cv2.ximgproc.segmentation.createSelectiveSearchSegmentationStrategyTexture()
    strategy_multiple = cv2.ximgproc.segmentation.createSelectiveSearchSegmentationStrategyMultiple(
        strategy_color,strategy_fill, strategy_size, strategy_texture)
    ss.addStrategy(strategy_size)
    return ss

def maskcut_forward(feats, dims, scales, init_image_size, tau=0, N=3, cpu=False):
    bipartitions = []
    eigvecs = []

    for i in range(N):
        if i == 0:
            painting = torch.from_numpy(np.zeros(dims))
            if not cpu: painting = painting.cuda()
        else:
            feats, painting = get_masked_affinity_matrix(painting, feats, current_mask, ps)

        # construct the affinity matrix
        A, D = get_affinity_matrix(feats, tau)
        # get the second smallest eigenvector
        eigenvec, second_smallest_vec = second_smallest_eigenvector(A, D)
        # get salient area
        bipartition = get_salient_areas(second_smallest_vec)
        seed = np.argmax(np.abs(second_smallest_vec))
        nc = check_num_fg_corners(bipartition, dims)
        if nc >= 3:
            reverse = True
        else:
            reverse = bipartition[seed] != 1

        if reverse:
            # reverse bipartition, eigenvector and get new seed
            eigenvec = eigenvec * -1
            bipartition = np.logical_not(bipartition)
            seed = np.argmax(eigenvec)
        else:
            seed = np.argmax(second_smallest_vec)

        # get pxiels corresponding to the seed
        bipartition = bipartition.reshape(dims).astype(float)
        _, _, _, cc = detect_box(bipartition, seed, dims, scales=scales, initial_im_size=init_image_size)
        pseudo_mask = np.zeros(dims)
        pseudo_mask[cc[0],cc[1]] = 1
        pseudo_mask = torch.from_numpy(pseudo_mask)
        if not cpu: pseudo_mask = pseudo_mask.to('cuda')
        ps = pseudo_mask.shape[0]

        # check if the extra mask is heavily overlapped with the previous one or is too small.
        if i >= 1:
            ratio = torch.sum(pseudo_mask) / pseudo_mask.size()[0] / pseudo_mask.size()[1]
            if metric.IoU(current_mask, pseudo_mask) > 0.5 or ratio <= 0.01:
                pseudo_mask = np.zeros(dims)
                pseudo_mask = torch.from_numpy(pseudo_mask)
                if not cpu: pseudo_mask = pseudo_mask.to('cuda')
        current_mask = pseudo_mask

        # mask out foreground areas in previous stages
        masked_out = 0 if len(bipartitions) == 0 else np.sum(bipartitions, axis=0)
        bipartition = F.interpolate(pseudo_mask.unsqueeze(0).unsqueeze(0), size=init_image_size).squeeze()

        bipartition_masked = bipartition.cpu().numpy() - masked_out
        bipartition_masked[bipartition_masked <= 0] = 0
        bipartitions.append(bipartition_masked)

        # unsample the eigenvec
        eigvec = second_smallest_vec.reshape(dims)
        eigvec = torch.from_numpy(eigvec)
        if not cpu: eigvec = eigvec.to('cuda')
        eigvec = F.interpolate(eigvec.unsqueeze(0).unsqueeze(0), size=init_image_size, mode='nearest').squeeze()
        eigvecs.append(eigvec.cpu().numpy())

    return seed, bipartitions, eigvecs
class ClipOutputTarget:
    def __init__(self, category):
        self.category = category
    def __call__(self, model_output):
        if len(model_output.shape) == 1:
            return model_output[self.category]
        return model_output[:, self.category]
def calculate_fft_energy(image):
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = np.abs(fshift) 
    energy = np.sum(magnitude_spectrum)
    return energy

def energy_difference_rate(image, mask):
    foreground = cv2.bitwise_and(image, image, mask=mask)
    background = cv2.bitwise_and(image, image, mask=cv2.bitwise_not(mask))
    
    FE = calculate_fft_energy(foreground)
    BE = calculate_fft_energy(background)
    
    EDR = (FE - BE) / (FE + BE) if (FE + BE) != 0 else 0
    
    return np.abs(EDR)
def fft_analysis(image_pil, mask_np):
    import math
    image = np.array(image_pil)
    
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = np.log(np.abs(fshift) + 1) 
    radius = 50  
    center = (magnitude_spectrum.shape[0] // 2, magnitude_spectrum.shape[1] // 2)
    magnitude_spectrum_uint8 = np.uint8(255 * (magnitude_spectrum / np.max(magnitude_spectrum))).copy()
    mask_circle = np.zeros_like(magnitude_spectrum_uint8).astype(np.uint8)
    cv2.circle(mask_circle, center, radius, 255, -1)


    mag_spectrum_polar = cv2.linearPolar(magnitude_spectrum, (magnitude_spectrum.shape[1]//2, magnitude_spectrum.shape[0]//2), magnitude_spectrum.shape[1]//2, cv2.WARP_FILL_OUTLIERS)


    angular_energy_distribution = np.sum(mag_spectrum_polar, axis=0) 
    dec = np.std(angular_energy_distribution) 
    dec1 =  math.log((dec*(( image.shape[0] *image.shape[1])** (1/3)))) if  dec  != 0 else float('inf')

    radial_energy_distribution = np.sum(mag_spectrum_polar, axis=1) 
    ccec= np.std(radial_energy_distribution)  
    ccec1 =math.log((ccec*((image.shape[0] * image.shape[1])** (1/3))))  if  ccec  != 0 else float('inf') 
    mask_array = np.where(mask_np != 0, 255, 0).astype(np.uint8)
    
    foreground_pixels = np.count_nonzero(mask_array)
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    masked_image = cv2.bitwise_and(image, image, mask=mask_array)

    f = np.fft.fft2(masked_image)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = np.log(np.abs(fshift) + 1) 
    radius = 50 
    center = (magnitude_spectrum.shape[0] // 2, magnitude_spectrum.shape[1] // 2)
    magnitude_spectrum_uint8 = np.uint8(255 * (magnitude_spectrum / np.max(magnitude_spectrum))).copy()
    mask_circle = np.zeros_like(magnitude_spectrum_uint8).astype(np.uint8)
    cv2.circle(mask_circle, center, radius, 255, -1)

    mag_spectrum_polar = cv2.linearPolar(magnitude_spectrum, (magnitude_spectrum.shape[1]//2, magnitude_spectrum.shape[0]//2), magnitude_spectrum.shape[1]//2, cv2.WARP_FILL_OUTLIERS)

    angular_energy_distribution = np.sum(mag_spectrum_polar, axis=0) 
    dec = np.std(angular_energy_distribution)
    dec = math.log(dec*((foreground_pixels)** (1/3)))if  dec  != 0 else float('inf') 

    radial_energy_distribution = np.sum(mag_spectrum_polar, axis=1) 
    ccec= np.std(radial_energy_distribution)
    ccec = math.log((ccec*((foreground_pixels)** (1/3))))  if  ccec  != 0 else float('inf')
    q = ccec/ccec1
    p = dec/dec1
    weight_dec = 0.5
    weight_ccec = 0.5
    total_score =  weight_dec * p + weight_ccec * q
    return total_score

def semantic_similarity_with_text(image_tensor, text,model):
    inputs_text = processor(text=text, return_tensors="pt", padding=True)
    inputs_image = processor(images=image_tensor, return_tensors="pt", padding=True)

    with torch.no_grad():
        text_features = model.get_text_features(**inputs_text)
        image_features = model.get_image_features(**inputs_image)

    sim = torch.nn.functional.cosine_similarity(text_features, image_features)

    return sim.item()
def maskcut(img_path, backbone,patch_size, model,processor,tau,model_clip,fg_text_features,bg_text_features,cam, N=1, fixed_size=480, cpu=False) :
    
    I = Image.open(img_path).convert('RGB')
    im = cv2.imread(img_path)
    w_shape,h_shape,_ = im.shape
    regions = []
    get_boxs= []
    args = {'lr': 0.0002, 'max_prop': 30, 'lr_backbone_names': ['backbone.0'],  'lr_backbone': 2e-05, 'lr_linear_proj_names': ['reference_points',   'sampling_offsets'], 'lr_linear_proj_mult': 0.1, 'batch_size': 4,     'weight_decay': 0.0001, 'epochs': 50, 'lr_drop': 40, 'lr_drop_epochs': None,    'clip_max_norm': 0.1, 'sgd': False, 'filter_pct': -1, 'with_box_refine':   False, 'two_stage': False, 'strategy': 'topk', 'obj_embedding_head':  'intermediate', 'frozen_weights': None, 'backbone': 'resnet50', 'dilation':  False, 'position_embedding': 'sine', 'position_embedding_scale': 6.283185307179586, 'num_feature_levels': 4, 'enc_layers': 6, 'dec_layers': 6,   'dim_feedforward': 1024, 'hidden_dim': 256, 'dropout': 0.1, 'nheads': 8,  'num_queries': 300, 'dec_n_points': 4, 'enc_n_points': 4, 'pretrain': '',    'load_backbone': 'swav', 'masks': False, 'aux_loss': True, 'set_cost_class':   2, 'set_cost_bbox': 5, 'set_cost_giou': 2, 'object_embedding_loss_coeff': 1,  'mask_loss_coef': 1, 'dice_loss_coef': 1, 'cls_loss_coef': 2,    'bbox_loss_coef': 5, 'giou_loss_coef': 2, 'focal_alpha': 0.25,     'dataset_file': 'coco', 'dataset': 'imagenet', 'data_root': 'data',     'coco_panoptic_path': None, 'remove_difficult': False, 'output_dir': '',    'cache_path': 'cache/ilsvrc/ss_box_cache', 'device': 'cuda', 'seed': 42,   'resume': '', 'eval_every': 1, 'start_epoch': 0, 'eval': False, 'viz':    False, 'num_workers': 2, 'cache_mode': False, 'object_embedding_loss': False,  'model': 'deformable_detr'}
    args = Namespace(**args)
    model_s, criterion, postprocessors = build_model(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_s = model_s.to(device)
    checkpoint = torch.hub.load_state_dict_from_url("https://github.com/amirbar/DETReg/releases/download/1.0.0/full_coco_finetune.pth", progress=True,  map_location=torch.device("cuda"))
    load_msg = model_s.load_state_dict(checkpoint['model'], strict=False)
    transforms = make_coco_transforms('val')
    im = Image.open(img_path)
    im_t, _ = transforms(im, None)
    im_t =im_t.unsqueeze(0)
    im_t = im_t.to(device)
    res = model_s(im_t)
    scores = torch.sigmoid(res['pred_logits'][..., 1])
    pred_boxes = res['pred_boxes']
    pred_boxes = pred_boxes.to("cpu")
    img_w, img_h = im.size
    pred_boxes_ = box_cxcywh_to_xyxy(pred_boxes) * torch.Tensor([img_w, img_h, img_w, img_h])
    scope = scores.argsort(descending = True).to("cpu") # sort by model confidence
    get_boxes = pred_boxes_[0, scope[0, :30]].to("cpu").detach().numpy() # pick top 3 proposals
    fg_text_features = fg_text_features.to('cuda')
    regions = []
    for  rect in get_boxes:
        xmin, ymin, xmax, ymax = rect
        xmin = int(max(xmin,0))
        ymin = int(max(ymin,0))
        xmax = int(min(img_w,xmax))
        ymax= int(min(img_h,ymax))
        regions.append([xmin, ymin, xmax, ymax])
    text_prompts = read_txt_file('ovcamo_cato.txt')
    images = []
    for region in regions:
        images.append(I.crop(region))
    inputs = processor(text =text_prompts  ,images=images[:15], return_tensors="pt", padding=True)
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)
    bipartitions, eigvecs ,bipart_region= [], [],[]
    region_zip = list(zip(regions,probs.detach().numpy(),images))
    I_news = []
    num = 0
    for (region,prob,image) in region_zip[:15]:
        sorted_indices = np.argsort(prob)[::-1] 
        maxx = sorted_indices[0]
        op_2_values = prob[sorted_indices[:2]] 
        if(maxx == 75 or prob.max().item()<0.7 or op_2_values[0]+op_2_values[1]<0.8):
            continue
        fixed_size1 = 360
        I_new = image.resize((int(fixed_size1), int(fixed_size1)), PIL.Image.LANCZOS)
        I_resize, w, h, feat_w, feat_h = utils.resize_pil(I_new, patch_size)
        tensor = ToTensor(I_resize).unsqueeze(0)
        if not cpu: tensor = tensor.cuda()
        feat = backbone(tensor)[0]
        _, bipartition, eigvec = maskcut_forward(feat, [feat_h, feat_w], [patch_size, patch_size], [h,w], tau, N=1, cpu=cpu)

        image_features, attn_weight_list = model_clip.encode_image(tensor, h, w)
        bg_features_temp = bg_text_features.to('cuda')
        fg_features_temp = fg_text_features[maxx].to('cuda').unsqueeze(0)
        text_features_temp = torch.cat([fg_features_temp, bg_features_temp], dim=0)
        input_tensor = [image_features, text_features_temp.to('cuda'), h, w]
        targets = [ClipOutputTarget(0)]
        #torch.cuda.empty_cache()
        grayscale_cam, logits_per_image, attn_weight_last = cam(input_tensor=input_tensor,
                                                                           targets=targets,
                                                                           target_size=None)  # (ori_width, ori_height))
        grayscale_cam = grayscale_cam[0, :]
        
        attn_weight_list.append(attn_weight_last)
        attn_weight = [aw[:, 1:, 1:] for aw in attn_weight_list]  # (b, hxw, hxw)
        attn_weight = torch.stack(attn_weight, dim=0)[-8:]
        attn_weight = torch.mean(attn_weight, dim=0)
        attn_weight = attn_weight[0].cpu().detach()
        attn_weight = attn_weight.float()
        
        box, cnt = scoremap2bbox(scoremap=grayscale_cam, threshold=0.4, multi_contour_eval=True)
        aff_mask = torch.zeros((grayscale_cam.shape[0],grayscale_cam.shape[1]))
        for i_ in range(cnt):
            x0_, y0_, x1_, y1_ = box[i_]
            aff_mask[y0_:y1_, x0_:x1_] = 1

        aff_mask = aff_mask.view(1,grayscale_cam.shape[0] * grayscale_cam.shape[1])
        aff_mat = attn_weight
        trans_mat = aff_mat / torch.sum(aff_mat, dim=0, keepdim=True)
        trans_mat = trans_mat / torch.sum(trans_mat, dim=1, keepdim=True)
        for _ in range(2):
            trans_mat = trans_mat / torch.sum(trans_mat, dim=0, keepdim=True)
            trans_mat = trans_mat / torch.sum(trans_mat, dim=1, keepdim=True)
        trans_mat = (trans_mat + trans_mat.transpose(1, 0)) / 2
        for _ in range(1):
            trans_mat = torch.matmul(trans_mat, trans_mat)
        trans_mat = trans_mat * aff_mask
        cam_to_refine = torch.FloatTensor(grayscale_cam)
        cam_to_refine = cam_to_refine.view(-1,1)
        cam_refined = torch.matmul(trans_mat, cam_to_refine).reshape(h //16, w // 16)
        cam_refined = cam_refined.cpu().numpy().astype(np.float32)
        
        cam_refined_highres = scale_cam_image([cam_refined], bipartition[0].shape)[0]
        aff_mask_masked = cam_refined_highres
        aff_mask_masked[aff_mask_masked <= 0] = 0
        heatmap = cv2.applyColorMap(np.uint8(255 * aff_mask_masked), cv2.COLORMAP_JET)
        num = num + 1
        from scipy.ndimage import gaussian_filter

        soft_mask = gaussian_filter(bipartition[0].astype(float), sigma=1)
        
        pseudo_mask = densecrf(np.array(I_new), bipartition[0])
        pseudo_mask = ndimage.binary_fill_holes(pseudo_mask>=0.5)
        # filter out the mask that have a very different pseudo-mask after the CRF
        mask1 = torch.from_numpy(bipartition[0])
        mask2 = torch.from_numpy(pseudo_mask)
        mask1 = mask1.cuda()
        mask2 = mask2.cuda()
        if metric.IoU(mask1, mask2) < 0.5:
            pseudo_mask = pseudo_mask * -1
        pseudo_mask[pseudo_mask < 0] = 0
        a = 1
        b = 0.4
        aff1 = a*aff_mask_masked + b*pseudo_mask
        for _ in range(10):
            mask_binary = (aff1 >= 0.6).astype(int)
            fft = fft_analysis(I_new,mask_binary)
            mask_np = np.where(mask_binary > 0, 255, 0).astype(np.uint8)
            mask_pil = Image.fromarray(mask_np).convert("L")
            inv_mask_pil = Image.fromarray(255 - mask_np).convert("L")
            image_rgba = I_resize.convert("RGBA")
            foreground_image = Image.new("RGBA", I_resize.size)
            background_image = Image.new("RGBA", I_resize.size)
            foreground_image.paste(image_rgba, mask=mask_pil)
            background_image.paste(image_rgba, mask=inv_mask_pil)
            sim_foreground = semantic_similarity_with_text(foreground_image, text_prompts[maxx],model)
            sim_background = semantic_similarity_with_text(background_image, text_prompts[maxx],model)
            sim = sim_foreground / sim_background
            #print(fft,sim)
            a = 0.9*a +0.1*a*(sim)
            b = 0.9*b+0.1*b*(fft)
            aff1 = a*aff_mask_masked + b*pseudo_mask
        #print("")
        bipartitions += [mask_binary,]
        #bipartitions += [bipartition[0],]
        eigvecs += eigvec
        for i in range(len(bipartition)):
            I_news.append(I_new)
            bipart_region += [region]
    I_new = I.resize((int(fixed_size), int(fixed_size)), PIL.Image.LANCZOS)
    I_resize, w, h, feat_w, feat_h = utils.resize_pil(I_new, patch_size)
    tensor = ToTensor(I_resize).unsqueeze(0)
    if not cpu: tensor = tensor.cuda()
    feat = backbone(tensor)[0]
    _, bipartition, eigvec = maskcut_forward(feat, [feat_h, feat_w], [patch_size, patch_size], [h,w], tau, N=N, cpu=cpu)
    bipartitions += bipartition
    for i in range(len(bipartition)):
        I_news.append(I_new)
        q =  list([0,0,h_shape-1,w_shape-1])
        bipart_region.append(q)
    return bipartitions, eigvecs,bipart_region, I_news
def resize_binary_mask(array, new_size):
    image = Image.fromarray(array.astype(np.uint8)*255)
    image = image.resize(new_size)
    return np.asarray(image).astype(np.bool_)

def close_contour(contour):
    if not np.array_equal(contour[0], contour[-1]):
        contour = np.vstack((contour, contour[0]))
    return contour
def is_mask_contained(mask1, mask2, containment_threshold=0.75):
    """
    检查 mask2 是否被包含在 mask1 中
    """
    intersection = np.logical_and(mask1, mask2)
    q = np.sum(mask2)
    if q == 0:
        return 0,0
    iou = np.sum(intersection) / q

    return iou >= containment_threshold,iou

def create_image_info(image_id, file_name, image_size, 
                      date_captured=datetime.datetime.utcnow().isoformat(' '),
                      license_id=1, coco_url="", flickr_url=""):
    image_info = {
            "id": image_id,
            "file_name": file_name,
            "width": image_size[0],
            "height": image_size[1],
            "date_captured": date_captured,
            "license": license_id,
            "coco_url": coco_url,
            "flickr_url": flickr_url
    }
    return image_info
def find_horizontal_lines_at_extremes(mask, threshold=80):
    height,weight = mask.shape
    rows_where_mask = np.where(mask.any(axis=1))[0]
    if rows_where_mask.size == 0:
        return False 

    top_row = rows_where_mask[0]
    bottom_row = rows_where_mask[-1]

    if top_row!=0: 
        if np.max(np.convolve(mask[top_row], np.ones(threshold, dtype=int), 'valid')) == threshold:
            return True

    if bottom_row!=height-1: 
        if np.max(np.convolve(mask[bottom_row], np.ones(threshold, dtype=int), 'valid')) == threshold:
            return True

    cols_where_mask = np.where(mask.any(axis=0))[0]
    if cols_where_mask.size == 0:
        return False 

    left_col = cols_where_mask[0]
    right_col = cols_where_mask[-1]
    if left_col!=0:
        if np.max(np.convolve(mask[:, left_col], np.ones(threshold, dtype=int), 'valid')) == threshold:
            return True

    if right_col!=weight-1:
        if np.max(np.convolve(mask[:, right_col], np.ones(threshold, dtype=int), 'valid')) == threshold:
            return True

    return False


def create_annotation_info(annotation_id, image_id, category_info, binary_mask, 
                           image_size=None, bounding_box=None):
    upper = np.max(binary_mask)
    lower = np.min(binary_mask)
    thresh = upper / 2.0
    binary_mask[binary_mask > thresh] = upper
    binary_mask[binary_mask <= thresh] = lower
    if image_size is not None:
        binary_mask = resize_binary_mask(binary_mask.astype(np.uint8), image_size)
    #print(type(mask))
    binary_mask_encoded = encode(np.asfortranarray(binary_mask.astype(np.uint8)))

    area = mask.area(binary_mask_encoded)
    if area < 1:
        return None

    if bounding_box is None:
        bounding_box = mask.toBbox(binary_mask_encoded)

    rle = mask_util.encode(np.array(binary_mask[...,None], order="F", dtype="uint8"))[0]
    rle['counts'] = rle['counts'].decode('ascii')
    segmentation = rle

    annotation_info = {
        "id": annotation_id,
        "image_id": image_id,
        "category_id": category_info["id"],
        "iscrowd": 0,
        "area": area.tolist(),
        "bbox": bounding_box.tolist(),
        "segmentation": segmentation,
        "width": binary_mask.shape[0],
        "height": binary_mask.shape[1],
    } 

    return annotation_info

# necessay info used for coco style annotations
INFO = {
    "description": "ImageNet-1K: pseudo-masks with MaskCut",
    "url": "https://github.com/facebookresearch/CutLER",
    "version": "1.0",
    "year": 2024,
    "contributor": "Zhentao He",
    "date_created": datetime.datetime.utcnow().isoformat(' ')
}

LICENSES = [
    {
        "id": 1,
        "name": "Apache License",
        "url": "https://github.com/facebookresearch/CutLER/blob/main/LICENSE"
    }
]

# only one class, i.e. foreground
CATEGORIES = [
    {
        'id': 1,
        'name': 'fg',
        'supercategory': 'fg',
    },
]

convert = lambda text: int(text) if text.isdigit() else text.lower()
natrual_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]

output = {
        "info": INFO,
        "licenses": LICENSES,
        "categories": CATEGORIES,
        "images": [],
        "annotations": []}

category_info = {
    "is_crowd": 0,
    "id": 1
}
def calculate_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2)
    union = np.sum(np.logical_or(mask1, mask2))
    if union == 0:
        return 0
    iou = np.sum(intersection) / union
    return iou
def zeroshot_classifier( model,bg=False):
    with torch.no_grad():
        zeroshot_weights = []
        if bg:
            text_prompts=["a photo of complexe background"]
        else:
            text_prompts = read_txt_file('ovcamo_cato.txt')
        for texts in text_prompts:
            texts = clip.tokenize(texts).to('cuda') #tokenize
            class_embeddings = model.encode_text(texts) #embed with text encoder
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to('cuda')
    return zeroshot_weights.t()
def reshape_transform(tensor, height=28, width=28):
    tensor = tensor.permute(1, 0, 2)
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result

if __name__ == "__main__":

    parser = argparse.ArgumentParser('MaskCut script')
    # default arguments
    parser.add_argument('--out-dir', type=str, help='output directory')
    parser.add_argument('--vit-arch', type=str, default='small', choices=['base', 'small'], help='which architecture')
    parser.add_argument('--vit-feat', type=str, default='k', choices=['k', 'q', 'v', 'kqv'], help='which features')
    parser.add_argument('--patch-size', type=int, default=16, choices=[16, 8], help='patch size')
    parser.add_argument('--nb-vis', type=int, default=20, choices=[1, 200], help='nb of visualization')
    parser.add_argument('--img-path', type=str, default=None, help='single image visualization')

    # additional arguments
    parser.add_argument('--dataset-path', type=str, default="imagenet/train/", help='path to the dataset')
    parser.add_argument('--tau', type=float, default=0.2, help='threshold used for producing binary graph')
    parser.add_argument('--fixed_size', type=int, default=480, help='rescale the input images to a fixed size')
    parser.add_argument('--pretrain_path', type=str, default=None, help='path to pretrained model')
    parser.add_argument('--N', type=int, default=3, help='the maximum number of pseudo-masks per image')
    parser.add_argument('--cpu', action='store_true', help='use cpu')

    args = parser.parse_args()

    if args.pretrain_path is not None:
        url = args.pretrain_path
    if args.vit_arch == 'base' and args.patch_size == 8:
        if args.pretrain_path is None:
            url = "https://dl.fbaipublicfiles.com/dino/dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth"
        feat_dim = 768
    elif args.vit_arch == 'small' and args.patch_size == 8:
        if args.pretrain_path is None:
            url = "https://dl.fbaipublicfiles.com/dino/dino_deitsmall8_300ep_pretrain/dino_deitsmall8_300ep_pretrain.pth"
        feat_dim = 384

    backbone = dino.ViTFeat(url, feat_dim, args.vit_arch, args.vit_feat, args.patch_size)

    msg = 'Load {} pre-trained feature...'.format(args.vit_arch)
    backbone.eval()
    if not args.cpu:
        backbone.cuda()

    img_folders = os.listdir(args.dataset_path)

    if args.out_dir is not None and not os.path.exists(args.out_dir) :
        os.mkdir(args.out_dir)


    image_id, segmentation_id = 1, 1
    image_names = []
    
    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    model_clip, _ = clip.load("ViT-B-16.pt", device='cuda')
    fg_text_features = zeroshot_classifier( model_clip)#['a rendering of a weird {}.'], model)
    bg_text_features = zeroshot_classifier( model_clip,bg=True)#['a rendering of a weird {}.'], model)
    target_layers = [model_clip.visual.transformer.resblocks[-1].ln_1]
    cam = GradCAM(model=model_clip, target_layers=target_layers, reshape_transform=reshape_transform)
    for img_folder in img_folders:
        args.img_dir = os.path.join(args.dataset_path, img_folder)
        img_list = sorted(os.listdir(args.img_dir))
        for img_name in tqdm(img_list) :
            img_path = os.path.join(args.img_dir, img_name)
            bipartitions, _ ,regions,I_news = maskcut(img_path, backbone, args.patch_size, model,processor,args.tau,model_clip,fg_text_features,bg_text_features,cam, N=args.N, fixed_size=args.fixed_size,cpu=args.cpu)

            I = Image.open(img_path).convert('RGB')
            I_image = np.array(I)
            width, height = I.size
            segmentation_mask = np.zeros((height,width), dtype=np.uint8)
            pseudo_masks = []
            for idx, (bipartition,region,I_new) in enumerate(zip(bipartitions,regions,I_news)):
                segmentation_mask = np.zeros((height,width), dtype=np.uint8)
                pseudo_mask = densecrf(np.array(I_new), bipartition)
                pseudo_mask = ndimage.binary_fill_holes(pseudo_mask>=0.5)
                mask1 = torch.from_numpy(bipartition)
                mask2 = torch.from_numpy(pseudo_mask)
                if not args.cpu: 
                    mask1 = mask1.cuda()
                    mask2 = mask2.cuda()
                if metric.IoU(mask1, mask2) < 0.5:
                    pseudo_mask = pseudo_mask * -1
                pseudo_mask[pseudo_mask < 0] = 0
                pseudo_mask = Image.fromarray(np.uint8(pseudo_mask*255))
                pseudo_mask = np.asarray(pseudo_mask.resize((region[2]-region[0],region[3]-region[1])))
                pseudo_mask_all = np.zeros((height,width),dtype=np.uint8)
                pseudo_mask_all[region[1]:region[3],region[0]:region[2]]=pseudo_mask
                pseudo_mask = Image.fromarray(pseudo_mask_all)
                pseudo_mask = np.asarray(pseudo_mask,dtype=np.uint8)
                p = pseudo_mask.copy()
                upper = np.max(p)
                lower = np.min(p)
                thresh = upper / 2.0
                p[p > thresh] = upper
                p[p <= thresh] = lower
                import math
                result = find_horizontal_lines_at_extremes(p>0,int(math.sqrt(height*height+width*width)/8))
                if result:
                    continue
                pseudo_masks.append(p)
            filtered_masks = pseudo_masks.copy()
            for i,masks in enumerate(pseudo_masks):
                for j,other_mask in enumerate(pseudo_masks):
                    if i!=j:
                        iou = calculate_iou(masks>0, other_mask>0)
                        iconta,iou1= is_mask_contained(other_mask>0,masks>0)
                        if iou > 0.5 or iconta :
                            if np.sum(masks) < np.sum(other_mask):
                                filtered_masks[i] = np.zeros_like(masks)
            filtered_masks = [masks for masks in filtered_masks if np.sum(masks) > 0]
            for idx,pseudo_mask in enumerate(filtered_masks):
                if img_name not in image_names:
                    image_info = create_image_info(
                        image_id, "{}/{}".format(img_folder, img_name), (height, width, 3))
                    output["images"].append(image_info)
                    image_names.append(img_name)           
                annotation_info = create_annotation_info(
                    segmentation_id, image_id, category_info, pseudo_mask.astype(np.uint8), None)
                if annotation_info is not None:
                    output["annotations"].append(annotation_info)
                    segmentation_id += 1
            image_id += 1
    # save annotations
    json_name = '{}/imagenet_train_fixsize{}_tau{}_N{}.json'.format(args.out_dir, args.fixed_size, args.tau, args.N)
    with open(json_name, 'w') as output_json_file:
        json.dump(output, output_json_file)
    print(f'dumping {json_name}')
    print("Done: {} images; {} anns.".format(len(output['images']), len(output['annotations'])))
