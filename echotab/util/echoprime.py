import os
from typing import Dict, Iterable, List, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from decord import VideoReader, cpu

VIEW_TYPES = [
    'A2C',  # 0
    'A3C',
    'A4C',
    'A5C',
    'Apical_Doppler',
    'Doppler_Parasternal_Long',
    'Doppler_Parasternal_Short',
    'Parasternal_Long',
    'Parasternal_Short',
    'SSN',
    'Subcostal',
]
NUM_VIEWS = len(VIEW_TYPES)

ECHOPRIME_MEAN = torch.tensor([29.110628, 28.076836, 29.096405]).reshape(3, 1, 1, 1)
ECHOPRIME_STD = torch.tensor([47.989223, 46.456997, 47.20083]).reshape(3, 1, 1, 1)

# EchoPrime (mvit_v2_s) final patch grid: 8 temporal x 7 x 7 spatial -> 392 local tokens per view.
# Each patch spans 2 raw frames x 32 x 32 pixels (effective after multi-scale pooling).
_TEMPORAL_STRIDE = 2
_SPATIAL_PATCH = 32


def crop_and_scale(img, res=(224, 224), interpolation=cv2.INTER_CUBIC, zoom=0.1):
    in_res = (img.shape[1], img.shape[0])
    r_in = in_res[0] / in_res[1]
    r_out = res[0] / res[1]

    if r_in > r_out:
        padding = int(round((in_res[0] - r_out * in_res[1]) / 2))
        img = img[:, padding:-padding]
    if r_in < r_out:
        padding = int(round((in_res[1] - in_res[0] / r_out) / 2))
        img = img[padding:-padding]
    if zoom != 0:
        pad_x = round(int(img.shape[1] * zoom))
        pad_y = round(int(img.shape[0] * zoom))
        img = img[pad_y:-pad_y, pad_x:-pad_x]

    img = cv2.resize(img, res, interpolation=interpolation)
    return img


def load_video_frames(fname: str, sampling_rate: int = 2, clip_len: int = 16) -> np.ndarray:
    """Read `clip_len` frames (every `sampling_rate`-th frame, starting at frame 0) from
    a video. Clips shorter than `clip_len` are zero-padded at the end.

    Returns (T, H, W, C) uint8.
    """
    if not os.path.exists(fname):
        raise FileNotFoundError(fname)

    vr = VideoReader(fname, num_threads=1, ctx=cpu(0))
    frame_ids = list(range(0, len(vr), sampling_rate))[:clip_len]

    vr.seek(0)
    buffer = vr.get_batch(frame_ids).asnumpy()  # (T, H, W, C)

    T, H, W, C = buffer.shape
    if T < clip_len:
        padding = np.zeros((clip_len - T, H, W, C), dtype=buffer.dtype)
        buffer = np.concatenate((buffer, padding), axis=0)
    return buffer


def load_echoprime_clip(
    fname: str,
    sampling_rate: int = 2,
    num_frames: int = 16,
    img_size: int = 224,
) -> torch.Tensor:
    """Load one echo clip and preprocess it for EchoPrime.

    Returns (T, H, W, C) float32, standardised with the EchoPrime mean/std.
    """
    frames = load_video_frames(fname, sampling_rate, num_frames)  # (T, H, W, C) uint8

    scaled = np.zeros((len(frames), img_size, img_size, frames.shape[-1]), dtype=np.float32)
    for i, frame in enumerate(frames):
        scaled[i] = crop_and_scale(frame, res=(img_size, img_size))

    video = torch.from_numpy(scaled).permute(3, 0, 1, 2)  # (C, T, H, W) for broadcasting
    video = (video - ECHOPRIME_MEAN) / ECHOPRIME_STD
    return video.permute(1, 2, 3, 0)  # (T, H, W, C)


def load_study(
    paths: List[str],
    view_classifier: nn.Module,
    excluded_view_types: Iterable[str] = (),
) -> Tuple[Dict[str, torch.Tensor], pd.DataFrame]:
    """Load all clips of a study and assign them to view types with the view classifier.

    Each clip is assigned to its most likely view type. If several clips are assigned to
    the same view type, the one with the highest classifier confidence is used (the first
    one on ties). Clips assigned to an excluded view type (cfg.dataset.echo.excluded_view_types,
    i.e. views the model was not trained on) are not used.

    Returns:
        study:       view_type -> (1, T, H, W, C), the input of EchoTabEncoder
        assignments: one row per clip with columns path, view, confidence, selected
    """
    excluded_view_types = set(excluded_view_types)
    best: Dict[str, Tuple[float, int, torch.Tensor]] = {}  # view_type -> (confidence, row, clip)
    rows = []
    for path in paths:
        clip = load_echoprime_clip(path)
        confidence, view_idx = view_classifier(clip.unsqueeze(0))[0].max(dim=-1)
        view, confidence = VIEW_TYPES[view_idx.item()], confidence.item()
        rows.append({'path': path, 'view': view, 'confidence': confidence, 'selected': False})

        if view not in excluded_view_types and (view not in best or confidence > best[view][0]):
            best[view] = (confidence, len(rows) - 1, clip)

    study = {}
    for view in VIEW_TYPES:  # canonical view order
        if view in best:
            _, row, clip = best[view]
            rows[row]['selected'] = True
            study[view] = clip.unsqueeze(0)
    return study, pd.DataFrame(rows, columns=['path', 'view', 'confidence', 'selected'])
