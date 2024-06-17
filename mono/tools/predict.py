import os
import os.path as osp
import cv2
import time
import sys

CODE_SPACE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(CODE_SPACE)
import argparse
import mmcv
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

try:
    from mmcv.utils import Config, DictAction
except:
    from mmengine import Config, DictAction
from datetime import timedelta
import random
import numpy as np
from mono.utils.logger import setup_logger
import glob
from mono.utils.comm import init_env
from mono.model.monodepth_model import get_configured_monodepth_model
from mono.utils.running import load_ckpt
from mono.utils.do_test import do_scalecano_test_with_custom_data
from mono.utils.mldb import load_data_info, reset_ckpt_path

import matplotlib.pyplot


def parse_args():
    parser = argparse.ArgumentParser(description='Train a segmentor')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--show-dir', help='the dir to save logs and visualization results')
    parser.add_argument('--load-from', help='the checkpoint file to load weights from')
    parser.add_argument('--node_rank', type=int, default=0)
    parser.add_argument('--nnodes', type=int, default=1, help='number of nodes')
    parser.add_argument('--options', nargs='+', action=DictAction, help='custom options')
    parser.add_argument('--launcher', choices=['None', 'pytorch', 'slurm', 'mpi', 'ror'], default='slurm',
                        help='job launcher'
                        )
    parser.add_argument('--test_data_path', default='None', type=str, help='the path of test data')
    parser.add_argument('--batch_size', default=1, type=int, help='the batch size for inference')
    args = parser.parse_args()
    return args


def resize_for_input(image, output_shape, canonical_shape, to_canonical_ratio):
    """
    Resize the input.
    Resizing consists of two processed, i.e. 1) to the canonical space (adjust the camera model); 2) resize the image while the camera model holds. Thus the
    label will be scaled with the resize factor.
    """
    padding = [123.675, 116.28, 103.53]
    h, w, _ = image.shape
    resize_ratio_h = output_shape[0] / canonical_shape[0]
    resize_ratio_w = output_shape[1] / canonical_shape[1]
    to_scale_ratio = min(resize_ratio_h, resize_ratio_w)
    
    resize_ratio = to_canonical_ratio * to_scale_ratio
    
    reshape_h = int(resize_ratio * h)
    reshape_w = int(resize_ratio * w)
    
    pad_h = max(output_shape[0] - reshape_h, 0)
    pad_w = max(output_shape[1] - reshape_w, 0)
    pad_h_half = int(pad_h / 2)
    pad_w_half = int(pad_w / 2)
    
    # resize
    image = cv2.resize(image, dsize=(reshape_w, reshape_h), interpolation=cv2.INTER_LINEAR)
    # padding
    image = cv2.copyMakeBorder(
            image,
            pad_h_half,
            pad_h - pad_h_half,
            pad_w_half,
            pad_w - pad_w_half,
            cv2.BORDER_CONSTANT,
            value=padding
    )
    
    pad = [pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half]
    label_scale_factor = 1 / to_scale_ratio
    
    return image, pad, label_scale_factor


def transform_test_data_scalecano(rgb, intrinsic, data_basic):
    """
    Pre-process the input for forwarding. Employ `label scale canonical transformation.'
        Args:
            rgb: input rgb image. [H, W, 3]
            intrinsic: camera intrinsic parameter, [fx, fy, u0, v0]
            data_basic: predefined canonical space in configs.
    """
    canonical_space = data_basic['canonical_space']
    forward_size = data_basic.crop_size
    mean = torch.tensor([123.675, 116.28, 103.53]).float()[:, None, None]
    std = torch.tensor([58.395, 57.12, 57.375]).float()[:, None, None]
    
    # BGR to RGB
    # rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    
    ori_h, ori_w, _ = rgb.shape
    ori_focal = (intrinsic[0] + intrinsic[1]) / 2
    canonical_focal = canonical_space['focal_length']
    
    cano_label_scale_ratio = canonical_focal / ori_focal
    
    # resize
    rgb, pad, resize_label_scale_ratio = resize_for_input(rgb, forward_size,
                                                          [ori_h, ori_w], 1.0
                                                          )
    
    # label scale factor
    label_scale_factor = cano_label_scale_ratio * resize_label_scale_ratio
    
    rgb = torch.from_numpy(rgb.transpose((2, 0, 1))).float()
    rgb = torch.div((rgb - mean), std)
    rgb = rgb.cuda()
    
    return rgb, pad, label_scale_factor


def get_prediction(
        model: torch.nn.Module,
        input: torch.tensor,
        cam_model: torch.tensor,
):
    data = dict(
            input=input,
            cam_model=cam_model,
    )
    pred_depth, confidence, output_dict = model.module.inference(data)
    
    return pred_depth, output_dict


def gray_to_colormap(img, cmap='rainbow'):
    """
    Transfer gray map to matplotlib colormap
    """
    assert img.ndim == 2
    
    img[img < 0] = 0
    mask_invalid = img < 1e-10
    img = img / (img.max() + 1e-8)
    norm = matplotlib.colors.Normalize(vmin=0, vmax=1.1)
    cmap_m = matplotlib.pyplot.get_cmap(cmap)
    map = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap_m)
    colormap = (map.to_rgba(img)[:, :, :3] * 255).astype(np.uint8)
    colormap[mask_invalid] = 0
    
    return colormap


def postprocess(pred_depth, rgb_origin, pad, normalize_scale, scale_info):
    pred_depth = pred_depth.squeeze()
    pred_depth = pred_depth[pad[0]: pred_depth.shape[0] - pad[1], pad[2]: pred_depth.shape[1] - pad[3]]
    pred_depth = torch.nn.functional.interpolate(pred_depth[None, None, :, :],
                                                 [rgb_origin.shape[0], rgb_origin.shape[1]], mode='bilinear'
                                                 ).squeeze()  # to original size
    pred_depth = pred_depth * normalize_scale / scale_info
    
    pred_depth = (pred_depth > 0) * (pred_depth < 300) * pred_depth
    
    rgb_torch = torch.from_numpy(rgb_origin).to(pred_depth.device).permute(2, 0, 1)
    mean = torch.tensor([123.675, 116.28, 103.53]).float()[:, None, None].to(rgb_torch.device)
    std = torch.tensor([58.395, 57.12, 57.375]).float()[:, None, None].to(rgb_torch.device)
    rgb_torch = torch.div((rgb_torch - mean), std)
    
    # pcd
    pred_depth = pred_depth.detach().cpu().numpy()
    pred_color = gray_to_colormap(pred_depth)
    pred_color = cv2.resize(pred_color, (rgb_origin.shape[1], rgb_origin.shape[0]))
    cv2.imshow("test", pred_color)
    
    # if an['intrinsic'] == None:
    #     # for r in [0.9, 1.0, 1.1]:
    #     for r in [1.0]:
    #         # for f in [600, 800, 1000, 1250, 1500]:
    #         for f in [1000]:
    #             pcd = reconstruct_pcd(pred_depth, f * r, f * (2 - r), intrinsic[2], intrinsic[3])
    #             fstr = '_fx_' + str(int(f * r)) + '_fy_' + str(int(f * (2 - r)))
    #             os.makedirs(osp.join(save_pcd_dir, an['folder']), exist_ok=True)
    #             save_point_cloud(pcd.reshape((-1, 3)), rgb_origin.reshape(-1, 3),
    #                              osp.join(save_pcd_dir, an['folder'], an['filename'][:-4] + fstr + '.ply')
    #                              )


def prediction(model: torch.nn.Module, rgb_origin: np.ndarray, cfg: dict):
    intrinsic = [1126.6, 1130.7, 977.7, 571.0]
    normalize_scale = cfg.data_basic.depth_range[1]
    
    # Pre-processing
    rgb_input, pad, label_scale_factor = transform_test_data_scalecano(rgb_origin, intrinsic, cfg.data_basic)
    
    pred_depths, outputs = get_prediction(
            model=model,
            input=torch.stack([rgb_input]),  # Stack inputs for batch processing
            cam_model=None
    )
    
    postprocess(
            pred_depths,
            rgb_origin,
            pad,
            normalize_scale,
            label_scale_factor,
    )


def main(args):
    os.chdir(CODE_SPACE)
    cfg = Config.fromfile(args.config)
    
    if args.options is not None:
        cfg.merge_from_dict(args.options)
    
    # show_dir is determined in this priority: CLI > segment in file > filename
    if args.show_dir is not None:
        # update configs according to CLI args if args.show_dir is not None
        cfg.show_dir = args.show_dir
    else:
        # use condig filename + timestamp as default show_dir if args.show_dir is None
        cfg.show_dir = osp.join('./show_dirs',
                                osp.splitext(osp.basename(args.config))[0],
                                args.timestamp
                                )
    
    # ckpt path
    if args.load_from is None:
        raise RuntimeError('Please set model path!')
    cfg.load_from = args.load_from
    cfg.batch_size = args.batch_size
    
    # load data info
    data_info = {}
    load_data_info('data_info', data_info=data_info)
    cfg.mldb_info = data_info
    # update check point info
    reset_ckpt_path(cfg.model, data_info)
    
    # create show dir
    os.makedirs(osp.abspath(cfg.show_dir), exist_ok=True)
    
    # init the logger before other steps
    cfg.log_file = osp.join(cfg.show_dir, f'{args.timestamp}.log')
    logger = setup_logger(cfg.log_file)
    
    # log some basic info
    logger.info(f'Config:\n{cfg.pretty_text}')
    
    # dump config
    cfg.dump(osp.join(cfg.show_dir, osp.basename(args.config)))
    
    main_worker(cfg)


def main_worker(cfg: dict):
    # build model
    model = get_configured_monodepth_model(cfg)
    model = torch.nn.DataParallel(model).cuda()
    
    # load ckpt
    model, _, _, _ = load_ckpt(cfg.load_from, model, strict_match=False)
    model.eval()
    
    cap = cv2.VideoCapture(0)
    
    ret, frame = cap.read()
    while ret:
        prediction(
                model,
                frame,
                cfg
        )
        
        if cv2.waitKey(1) == 27:
            break
        
        ret, frame = cap.read()


if __name__ == '__main__':
    args = parse_args()
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    args.timestamp = timestamp
    main(args)
