dependencies = ['torch', 'torchvision']

import os
import torch
from ultralytics import YOLO
# import cv2 as cv
import pygame
import time
from enum import Enum
import numpy as np

try:
    from mmcv.utils import Config, DictAction
except:
    from mmengine import Config, DictAction

from mono.model.monodepth_model import get_configured_monodepth_model

metric3d_dir = os.path.dirname(__file__)

MODEL_TYPE = {
    'ConvNeXt-Tiny':  {
        'cfg_file':  f'{metric3d_dir}/mono/configs/HourglassDecoder/convtiny.0.3_150.py',
        'ckpt_file': 'https://huggingface.co/JUGGHM/Metric3D/resolve/main/convtiny_hourglass_v1.pth',
    },
    'ConvNeXt-Large': {
        'cfg_file':  f'{metric3d_dir}/mono/configs/HourglassDecoder/convlarge.0.3_150.py',
        'ckpt_file': 'https://huggingface.co/JUGGHM/Metric3D/resolve/main/convlarge_hourglass_0.3_150_step750k_v1.1.pth',
    },
    'ViT-Small':      {
        'cfg_file':  f'{metric3d_dir}/mono/configs/HourglassDecoder/vit.raft5.small.py',
        'ckpt_file': 'https://huggingface.co/JUGGHM/Metric3D/resolve/main/metric_depth_vit_small_800k.pth',
    },
    'ViT-Large':      {
        'cfg_file':  f'{metric3d_dir}/mono/configs/HourglassDecoder/vit.raft5.large.py',
        'ckpt_file': 'https://huggingface.co/JUGGHM/Metric3D/resolve/main/metric_depth_vit_large_800k.pth',
    },
    'ViT-giant2':     {
        'cfg_file':  f'{metric3d_dir}/mono/configs/HourglassDecoder/vit.raft5.giant2.py',
        'ckpt_file': 'https://huggingface.co/JUGGHM/Metric3D/resolve/main/metric_depth_vit_giant2_800k.pth',
    },
}


def metric3d_convnext_tiny(pretrain=False, **kwargs):
    '''
    Return a Metric3D model with ConvNeXt-Large backbone and Hourglass-Decoder head.
    For usage examples, refer to: https://github.com/YvanYin/Metric3D/blob/main/hubconf.py
    Args:
      pretrain (bool): whether to load pretrained weights.
    Returns:
      model (nn.Module): a Metric3D model.
    '''
    cfg_file = MODEL_TYPE['ConvNeXt-Tiny']['cfg_file']
    ckpt_file = MODEL_TYPE['ConvNeXt-Tiny']['ckpt_file']
    
    cfg = Config.fromfile(cfg_file)
    model = get_configured_monodepth_model(cfg)
    if pretrain:
        model.load_state_dict(
                torch.hub.load_state_dict_from_url(ckpt_file)['model_state_dict'],
                strict=False,
        )
    return model


def metric3d_convnext_large(pretrain=False, **kwargs):
    '''
    Return a Metric3D model with ConvNeXt-Large backbone and Hourglass-Decoder head.
    For usage examples, refer to: https://github.com/YvanYin/Metric3D/blob/main/hubconf.py
    Args:
      pretrain (bool): whether to load pretrained weights.
    Returns:
      model (nn.Module): a Metric3D model.
    '''
    cfg_file = MODEL_TYPE['ConvNeXt-Large']['cfg_file']
    ckpt_file = MODEL_TYPE['ConvNeXt-Large']['ckpt_file']
    
    cfg = Config.fromfile(cfg_file)
    model = get_configured_monodepth_model(cfg)
    if pretrain:
        model.load_state_dict(
                torch.hub.load_state_dict_from_url(ckpt_file)['model_state_dict'],
                strict=False,
        )
    return model


def metric3d_vit_small(pretrain=False, **kwargs):
    '''
    Return a Metric3D model with ViT-Small backbone and RAFT-4iter head.
    For usage examples, refer to: https://github.com/YvanYin/Metric3D/blob/main/hubconf.py
    Args:
      pretrain (bool): whether to load pretrained weights.
    Returns:
      model (nn.Module): a Metric3D model.
    '''
    cfg_file = MODEL_TYPE['ViT-Small']['cfg_file']
    ckpt_file = MODEL_TYPE['ViT-Small']['ckpt_file']
    
    cfg = Config.fromfile(cfg_file)
    model = get_configured_monodepth_model(cfg)
    if pretrain:
        model.load_state_dict(
                torch.hub.load_state_dict_from_url(ckpt_file)['model_state_dict'],
                strict=False,
        )
    return model


def metric3d_vit_large(pretrain=False, **kwargs):
    '''
    Return a Metric3D model with ViT-Large backbone and RAFT-8iter head.
    For usage examples, refer to: https://github.com/YvanYin/Metric3D/blob/main/hubconf.py
    Args:
      pretrain (bool): whether to load pretrained weights.
    Returns:
      model (nn.Module): a Metric3D model.
    '''
    cfg_file = MODEL_TYPE['ViT-Large']['cfg_file']
    ckpt_file = MODEL_TYPE['ViT-Large']['ckpt_file']
    
    cfg = Config.fromfile(cfg_file)
    model = get_configured_monodepth_model(cfg)
    if pretrain:
        model.load_state_dict(
                torch.hub.load_state_dict_from_url(ckpt_file)['model_state_dict'],
                strict=False,
        )
    return model


def metric3d_vit_giant2(pretrain=False, **kwargs):
    '''
    Return a Metric3D model with ViT-Giant2 backbone and RAFT-8iter head.
    For usage examples, refer to: https://github.com/YvanYin/Metric3D/blob/main/hubconf.py
    Args:
      pretrain (bool): whether to load pretrained weights.
    Returns:
      model (nn.Module): a Metric3D model.
    '''
    cfg_file = MODEL_TYPE['ViT-giant2']['cfg_file']
    ckpt_file = MODEL_TYPE['ViT-giant2']['ckpt_file']
    
    cfg = Config.fromfile(cfg_file)
    model = get_configured_monodepth_model(cfg)
    if pretrain:
        model.load_state_dict(
                torch.hub.load_state_dict_from_url(ckpt_file)['model_state_dict'],
                strict=False,
        )
    return model


def metric3d(model: torch.nn.Module, rgb_origin: np.ndarray) -> np.ndarray:
    intrinsic = [1126.6, 1130.7, 977.7, 571.0]
    
    #### ajust input size to fit pretrained model
    # keep ratio resize
    input_size = (616, 1064)  # for vit model
    # input_size = (544, 1216) # for convnext model
    h, w = rgb_origin.shape[:2]
    scale = min(input_size[0] / h, input_size[1] / w)
    rgb = cv2.resize(rgb_origin, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)
    # remember to scale intrinsic, hold depth
    intrinsic = [intrinsic[0] * scale, intrinsic[1] * scale, intrinsic[2] * scale, intrinsic[3] * scale]
    # padding to input_size
    padding = [123.675, 116.28, 103.53]
    h, w = rgb.shape[:2]
    pad_h = input_size[0] - h
    pad_w = input_size[1] - w
    pad_h_half = pad_h // 2
    pad_w_half = pad_w // 2
    rgb = cv2.copyMakeBorder(rgb, pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half, cv2.BORDER_CONSTANT,
                             value=padding
                             )
    pad_info = [pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half]
    
    #### normalize
    mean = torch.tensor([123.675, 116.28, 103.53]).float()[:, None, None]
    std = torch.tensor([58.395, 57.12, 57.375]).float()[:, None, None]
    rgb = torch.from_numpy(rgb.transpose((2, 0, 1))).float()
    rgb = torch.div((rgb - mean), std)
    rgb = rgb[None, :, :, :].cuda()
    
    ###################### canonical camera space ######################
    # inference
    model.cuda().eval()
    with torch.no_grad():
        pred_depth, confidence, output_dict = model.inference({'input': rgb})
    
    # un pad
    pred_depth = pred_depth.squeeze()
    pred_depth = pred_depth[pad_info[0]: pred_depth.shape[0] - pad_info[1],
                 pad_info[2]: pred_depth.shape[1] - pad_info[3]]
    
    # upsample to original size
    pred_depth = torch.nn.functional.interpolate(pred_depth[None, None, :, :], rgb_origin.shape[:2], mode='bilinear'
                                                 ).squeeze()
    ###################### canonical camera space ######################
    
    #### de-canonical transform
    canonical_to_real_scale = intrinsic[0] / 1000.0  # 1000.0 is the focal length of canonical camera
    pred_depth = pred_depth * canonical_to_real_scale  # now the depth is metric
    pred_depth = torch.clamp(pred_depth, 0, 300)
    
    #### normal are also available
    # if 'prediction_normal' in output_dict:  # only available for Metric3Dv2, i.e. vit model
    pred_normal = output_dict['prediction_normal'][:, :3, :, :]
    #     normal_confidence = output_dict['prediction_normal'][:, 3, :, :]  # see https://arxiv.org/abs/2109.09881 for details
    #     # un pad and resize to some size if needed
    pred_normal = pred_normal.squeeze()
    pred_normal = pred_normal[:, pad_info[0]: pred_normal.shape[1] - pad_info[1],
                      pad_info[2]: pred_normal.shape[2] - pad_info[3]]
    #     # you can now do anything with the normal
    #     # such as visualize pred_normal
    pred_normal_vis = pred_normal.cpu().numpy().transpose((1, 2, 0))
    pred_normal_vis = (pred_normal_vis + 1) / 2
    cv2.imshow('normal_vis.png', (pred_normal_vis * 255).astype(np.uint8))
        
    return pred_depth.cpu().numpy()

play_list = (("mixkit-arcade-chiptune-explosion-1691.wav"),
             ("mixkit-police-siren-1641.wav"))

class Status(Enum):
    NOK = 0
    YOK = 1
    FAR = 2
    CLOSE = 3

CLASS_NORTH_KOREAN = 77
CLASS_PERSON = 0
CLASS_BICYCLE = 1
CLASS_CAR = 2
CLASS_MOTORCYCLE = 3
CLASSES = [CLASS_PERSON, CLASS_BICYCLE, CLASS_CAR, CLASS_MOTORCYCLE, CLASS_NORTH_KOREAN]


if __name__ == '__main__':
    import cv2
    import numpy as np

    DISTANCE_THRESHOLD = 2.0
    pygame.init()
    model_detect = YOLO('yolov8s.pt')

    explosion_sound = pygame.mixer.Sound(play_list[0])
    siren_sound = pygame.mixer.Sound(play_list[1])

    cap = cv2.VideoCapture(0)
    
    ret, frame = cap.read()
    status_flag = Status.NOK
    model = torch.hub.load('yvanyin/metric3d', 'metric3d_vit_small', pretrain=True)
    
    while ret:
        depth = metric3d(model, frame)
        frame_detect = frame
        result_detect = model_detect.predict(frame,classes=CLASSES,half=True)

        cv2.imshow("Pure Frame", result_detect[0].plot())
        class_cpu = result_detect[0].boxes.cls.detach().cpu().numpy()
        bbox_coords_cpu = result_detect[0].boxes.xyxy.to('cpu').numpy()

        for bbox_idx in range(len(bbox_coords_cpu)):
            bbox = bbox_coords_cpu[bbox_idx]
            # If there is a vehicle in the image, the control department should give a warning(Siren)
            if class_cpu[bbox_idx] in CLASSES[1:3]:
                status_flag = Status.FAR
            elif class_cpu[bbox_idx] == CLASS_NORTH_KOREAN:
                print('North Korean invaded!')
                middle_point = (int((bbox[1] + bbox[3]) / 2),int((bbox[0] + bbox[2]) / 2))
                depth_middle = depth[middle_point]
                if(depth_middle>DISTANCE_THRESHOLD):
                    status_flag = Status.FAR
                else:
                    status_flag = Status.CLOSE
                print("Depth of the middle point",depth_middle)
                break

        if status_flag == Status.FAR:
            siren_sound.play()
            time.sleep(3)
            pygame.mixer.music.set_volume(1.0)
        elif status_flag == Status.CLOSE:
            explosion_sound.play()
            time.sleep(1)
            pygame.mixer.music.set_volume(1.0)

        status_flag = Status.NOK

        if cv2.waitKey(10) == 27:
            break
        ret, frame = cap.read()

    pygame.quit()
    cv2.destroyAllWindows()
    cap.release()