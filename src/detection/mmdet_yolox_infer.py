# Vendored HSG-SRL artifact copy. Isolated from the development repo;
# edit this file only. See src/SOURCE_MAP.csv.
"""Load and run MMDetection 2.x YOLOX checkpoints without installing mmdet/mmcv.

The Synthehicle ``yolox_synthehicle_*.pth`` weights are MMDet YOLOX-X checkpoints
(``mmcv_version`` 1.6.x, ``bbox_head.num_classes=1``). This module rebuilds the
YOLOX graph with matching ``state_dict`` key names and runs letterbox → decode → NMS.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2 as cv
import numpy as np
import torch
import torch.nn as nn
from torchvision.ops import batched_nms


# ── building blocks (MMDet ConvModule / CSP naming) ─────────────────────────


class ConvModule(nn.Module):
    """Conv + BN + SiLU with nested ``.conv`` / ``.bn`` attribute names."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        padding: int | None = None,
        bias: bool = False,
    ) -> None:
        super().__init__()
        if padding is None:
            padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.03)
        self.activate = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activate(self.bn(self.conv(x)))


class Focus(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 1) -> None:
        super().__init__()
        self.conv = ConvModule(in_channels * 4, out_channels, kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        patch_top_left = x[..., ::2, ::2]
        patch_top_right = x[..., ::2, 1::2]
        patch_bot_left = x[..., 1::2, ::2]
        patch_bot_right = x[..., 1::2, 1::2]
        x = torch.cat(
            (patch_top_left, patch_bot_left, patch_top_right, patch_bot_right),
            dim=1,
        )
        return self.conv(x)


class DarknetBottleneck(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        expansion: float = 0.5,
        add_identity: bool = True,
    ) -> None:
        super().__init__()
        hidden = int(out_channels * expansion)
        self.conv1 = ConvModule(in_channels, hidden, 1)
        self.conv2 = ConvModule(hidden, out_channels, 3, stride=1, padding=1)
        self.add_identity = add_identity and in_channels == out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv2(self.conv1(x))
        return x + out if self.add_identity else out


class CSPLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int = 1,
        add_identity: bool = True,
        expand_ratio: float = 0.5,
    ) -> None:
        super().__init__()
        mid = int(out_channels * expand_ratio)
        self.main_conv = ConvModule(in_channels, mid, 1)
        self.short_conv = ConvModule(in_channels, mid, 1)
        self.final_conv = ConvModule(2 * mid, out_channels, 1)
        self.blocks = nn.Sequential(
            *[
                DarknetBottleneck(mid, mid, 1.0, add_identity)
                for _ in range(num_blocks)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_short = self.short_conv(x)
        x_main = self.blocks(self.main_conv(x))
        return self.final_conv(torch.cat((x_main, x_short), dim=1))


class SPPBottleneck(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_sizes: tuple[int, ...] = (5, 9, 13),
    ) -> None:
        super().__init__()
        mid = in_channels // 2
        self.conv1 = ConvModule(in_channels, mid, 1)
        self.poolings = nn.ModuleList(
            [nn.MaxPool2d(kernel_size=ks, stride=1, padding=ks // 2) for ks in kernel_sizes]
        )
        self.conv2 = ConvModule(mid * (len(kernel_sizes) + 1), out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = torch.cat([x] + [pool(x) for pool in self.poolings], dim=1)
        return self.conv2(x)


class CSPDarknet(nn.Module):
    """MMDet CSPDarknet-P5 used by YOLOX."""

    arch_settings = {
        "P5": [
            [64, 128, 3, True, False],
            [128, 256, 9, True, False],
            [256, 512, 9, True, False],
            [512, 1024, 3, False, True],
        ]
    }

    def __init__(
        self,
        deepen_factor: float = 1.0,
        widen_factor: float = 1.0,
        out_indices: tuple[int, ...] = (2, 3, 4),
    ) -> None:
        super().__init__()
        self.out_indices = out_indices
        arch = self.arch_settings["P5"]
        self.stem = Focus(3, int(arch[0][0] * widen_factor), kernel_size=3)
        self.layers = ["stem"]
        for i, (in_c, out_c, n_blocks, add_identity, use_spp) in enumerate(arch):
            in_c = int(in_c * widen_factor)
            out_c = int(out_c * widen_factor)
            n_blocks = max(round(n_blocks * deepen_factor), 1)
            stage: list[nn.Module] = [
                ConvModule(in_c, out_c, 3, stride=2, padding=1)
            ]
            if use_spp:
                stage.append(SPPBottleneck(out_c, out_c))
            stage.append(
                CSPLayer(out_c, out_c, num_blocks=n_blocks, add_identity=add_identity)
            )
            self.add_module(f"stage{i + 1}", nn.Sequential(*stage))
            self.layers.append(f"stage{i + 1}")

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        outs = []
        for i, name in enumerate(self.layers):
            x = getattr(self, name)(x)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)


class YOLOXPAFPN(nn.Module):
    def __init__(
        self,
        in_channels: list[int],
        out_channels: int,
        num_csp_blocks: int = 3,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.reduce_layers = nn.ModuleList()
        self.top_down_blocks = nn.ModuleList()
        for idx in range(len(in_channels) - 1, 0, -1):
            self.reduce_layers.append(ConvModule(in_channels[idx], in_channels[idx - 1], 1))
            self.top_down_blocks.append(
                CSPLayer(
                    in_channels[idx - 1] * 2,
                    in_channels[idx - 1],
                    num_blocks=num_csp_blocks,
                    add_identity=False,
                )
            )
        self.downsamples = nn.ModuleList()
        self.bottom_up_blocks = nn.ModuleList()
        for idx in range(len(in_channels) - 1):
            self.downsamples.append(
                ConvModule(in_channels[idx], in_channels[idx], 3, stride=2, padding=1)
            )
            self.bottom_up_blocks.append(
                CSPLayer(
                    in_channels[idx] * 2,
                    in_channels[idx + 1],
                    num_blocks=num_csp_blocks,
                    add_identity=False,
                )
            )
        self.out_convs = nn.ModuleList(
            [ConvModule(in_channels[i], out_channels, 1) for i in range(len(in_channels))]
        )

    def forward(self, inputs: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
        inner_outs = [inputs[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            feat_high = self.reduce_layers[len(self.in_channels) - 1 - idx](inner_outs[0])
            inner_outs[0] = feat_high
            upsample_feat = self.upsample(feat_high)
            feat_low = inputs[idx - 1]
            inner_out = self.top_down_blocks[len(self.in_channels) - 1 - idx](
                torch.cat([upsample_feat, feat_low], 1)
            )
            inner_outs.insert(0, inner_out)

        outs = [inner_outs[0]]
        for idx in range(len(self.in_channels) - 1):
            downsample_feat = self.downsamples[idx](outs[-1])
            out = self.bottom_up_blocks[idx](
                torch.cat([downsample_feat, inner_outs[idx + 1]], 1)
            )
            outs.append(out)

        for idx, conv in enumerate(self.out_convs):
            outs[idx] = conv(outs[idx])
        return tuple(outs)


class YOLOXHead(nn.Module):
    def __init__(
        self,
        num_classes: int,
        in_channels: int,
        feat_channels: int = 256,
        stacked_convs: int = 2,
        strides: tuple[int, ...] = (8, 16, 32),
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.stacked_convs = stacked_convs
        self.strides = strides
        self.multi_level_cls_convs = nn.ModuleList()
        self.multi_level_reg_convs = nn.ModuleList()
        self.multi_level_conv_cls = nn.ModuleList()
        self.multi_level_conv_reg = nn.ModuleList()
        self.multi_level_conv_obj = nn.ModuleList()
        for _ in strides:
            self.multi_level_cls_convs.append(self._build_stacked_convs())
            self.multi_level_reg_convs.append(self._build_stacked_convs())
            self.multi_level_conv_cls.append(nn.Conv2d(feat_channels, num_classes, 1))
            self.multi_level_conv_reg.append(nn.Conv2d(feat_channels, 4, 1))
            self.multi_level_conv_obj.append(nn.Conv2d(feat_channels, 1, 1))

    def _build_stacked_convs(self) -> nn.Sequential:
        layers = []
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            layers.append(ConvModule(chn, self.feat_channels, 3, stride=1, padding=1))
        return nn.Sequential(*layers)

    def forward(
        self, feats: tuple[torch.Tensor, ...]
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        cls_scores, bbox_preds, objectnesses = [], [], []
        for feat, cls_convs, reg_convs, conv_cls, conv_reg, conv_obj in zip(
            feats,
            self.multi_level_cls_convs,
            self.multi_level_reg_convs,
            self.multi_level_conv_cls,
            self.multi_level_conv_reg,
            self.multi_level_conv_obj,
        ):
            cls_feat = cls_convs(feat)
            reg_feat = reg_convs(feat)
            cls_scores.append(conv_cls(cls_feat))
            bbox_preds.append(conv_reg(reg_feat))
            objectnesses.append(conv_obj(reg_feat))
        return cls_scores, bbox_preds, objectnesses


class YOLOX(nn.Module):
    def __init__(
        self,
        deepen_factor: float = 1.33,
        widen_factor: float = 1.25,
        num_classes: int = 1,
        in_channels: list[int] | None = None,
        out_channels: int = 320,
        num_csp_blocks: int = 4,
        feat_channels: int = 320,
    ) -> None:
        super().__init__()
        if in_channels is None:
            in_channels = [320, 640, 1280]
        self.backbone = CSPDarknet(deepen_factor=deepen_factor, widen_factor=widen_factor)
        self.neck = YOLOXPAFPN(in_channels, out_channels, num_csp_blocks=num_csp_blocks)
        self.bbox_head = YOLOXHead(
            num_classes=num_classes,
            in_channels=out_channels,
            feat_channels=feat_channels,
        )

    def forward(
        self, x: torch.Tensor
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        return self.bbox_head(self.neck(self.backbone(x)))


# ── checkpoint / preprocess / postprocess ───────────────────────────────────


@dataclass
class YoloxCheckpointMeta:
    deepen_factor: float = 1.33
    widen_factor: float = 1.25
    num_classes: int = 1
    in_channels: list[int] | None = None
    out_channels: int = 320
    num_csp_blocks: int = 4
    feat_channels: int = 320
    img_scale: tuple[int, int] = (640, 640)
    score_thr: float = 0.01
    nms_iou: float = 0.65


def _safe_literal_dict(text: str) -> dict[str, Any]:
    """Parse a single ``dict(...)`` fragment from a dumped mmcv config string."""
    try:
        value = ast.literal_eval(text)
        return value if isinstance(value, dict) else {}
    except (SyntaxError, ValueError):
        return {}


def parse_yolox_meta_from_config(config_text: str) -> YoloxCheckpointMeta:
    meta = YoloxCheckpointMeta()
    m = re.search(r"img_scale\s*=\s*(\([^)]+\))", config_text)
    if m:
        try:
            scale = ast.literal_eval(m.group(1))
            if isinstance(scale, (list, tuple)) and len(scale) == 2:
                meta.img_scale = (int(scale[0]), int(scale[1]))
        except (SyntaxError, ValueError, TypeError):
            pass

    m = re.search(r"model\s*=\s*(dict\([\s\S]*?\n\))", config_text)
    model_dict = _safe_literal_dict(m.group(1)) if m else {}
    if not model_dict:
        # Fallback: extract key fields with regex when literal_eval fails
        # (mmcv dumps often contain non-literal path strings earlier in file).
        dm = re.search(r"deepen_factor\s*=\s*([0-9.]+)", config_text)
        wm = re.search(r"widen_factor\s*=\s*([0-9.]+)", config_text)
        nm = re.search(r"num_classes\s*=\s*(\d+)", config_text)
        if dm:
            meta.deepen_factor = float(dm.group(1))
        if wm:
            meta.widen_factor = float(wm.group(1))
        if nm:
            meta.num_classes = int(nm.group(1))
        ic = re.search(r"in_channels\s*=\s*(\[[^\]]+\])", config_text)
        if ic:
            try:
                channels = ast.literal_eval(ic.group(1))
                if isinstance(channels, list) and channels:
                    meta.in_channels = [int(c) for c in channels]
            except (SyntaxError, ValueError, TypeError):
                pass
        oc = re.search(r"out_channels\s*=\s*(\d+)", config_text)
        if oc:
            meta.out_channels = int(oc.group(1))
            meta.feat_channels = int(oc.group(1))
        nc = re.search(r"num_csp_blocks\s*=\s*(\d+)", config_text)
        if nc:
            meta.num_csp_blocks = int(nc.group(1))
        st = re.search(r"score_thr\s*=\s*([0-9.]+)", config_text)
        if st:
            meta.score_thr = float(st.group(1))
        ni = re.search(r"iou_threshold\s*=\s*([0-9.]+)", config_text)
        if ni:
            meta.nms_iou = float(ni.group(1))
        return meta

    backbone = model_dict.get("backbone") or {}
    neck = model_dict.get("neck") or {}
    head = model_dict.get("bbox_head") or {}
    test_cfg = model_dict.get("test_cfg") or {}
    meta.deepen_factor = float(backbone.get("deepen_factor", meta.deepen_factor))
    meta.widen_factor = float(backbone.get("widen_factor", meta.widen_factor))
    meta.num_classes = int(head.get("num_classes", meta.num_classes))
    if neck.get("in_channels"):
        meta.in_channels = [int(c) for c in neck["in_channels"]]
    meta.out_channels = int(neck.get("out_channels", meta.out_channels))
    meta.num_csp_blocks = int(neck.get("num_csp_blocks", meta.num_csp_blocks))
    meta.feat_channels = int(head.get("feat_channels", meta.out_channels))
    meta.score_thr = float(test_cfg.get("score_thr", meta.score_thr))
    nms = test_cfg.get("nms") or {}
    meta.nms_iou = float(nms.get("iou_threshold", meta.nms_iou))
    return meta


def _strip_state_dict(raw: dict[str, Any]) -> dict[str, torch.Tensor]:
    """Keep non-EMA weights; drop optimizer / hook tensors."""
    out: dict[str, torch.Tensor] = {}
    for key, value in raw.items():
        if not torch.is_tensor(value):
            continue
        if key.startswith("ema_"):
            continue
        if key.startswith("module."):
            key = key[len("module.") :]
        out[key] = value
    return out


def build_yolox_from_checkpoint(
    checkpoint_path: str | Path,
    device: str | torch.device = "cpu",
) -> tuple[YOLOX, YoloxCheckpointMeta]:
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"YOLOX checkpoint not found: {path}")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict) or "state_dict" not in ckpt:
        raise ValueError(f"Not an MMDet checkpoint (missing state_dict): {path}")
    meta = YoloxCheckpointMeta()
    cfg_text = (ckpt.get("meta") or {}).get("config")
    if isinstance(cfg_text, str) and cfg_text.strip():
        meta = parse_yolox_meta_from_config(cfg_text)

    model = YOLOX(
        deepen_factor=meta.deepen_factor,
        widen_factor=meta.widen_factor,
        num_classes=meta.num_classes,
        in_channels=meta.in_channels,
        out_channels=meta.out_channels,
        num_csp_blocks=meta.num_csp_blocks,
        feat_channels=meta.feat_channels,
    )
    state = _strip_state_dict(ckpt["state_dict"])
    missing, unexpected = model.load_state_dict(state, strict=False)
    # Ignore num_batches_tracked mismatches; fail on real structural misses.
    structural_missing = [
        k for k in missing if not k.endswith("num_batches_tracked")
    ]
    if structural_missing:
        preview = ", ".join(structural_missing[:8])
        raise RuntimeError(
            f"YOLOX weights do not match the network; missing "
            f"{len(structural_missing)} keys (example: {preview})"
        )
    if unexpected:
        # Allow unused keys (e.g. training-only), but warn via print.
        print(f"  [mmdet_yolox] ignoring unexpected keys: {len(unexpected)}")

    model.to(device)
    model.eval()
    return model, meta


def letterbox_square(
    image_bgr: np.ndarray,
    img_scale: tuple[int, int] = (640, 640),
    pad_val: float = 114.0,
) -> tuple[np.ndarray, float, tuple[int, int]]:
    """MMDet-style Resize(keep_ratio)+Pad(pad_to_square) for YOLOX."""
    target_h, target_w = int(img_scale[0]), int(img_scale[1])
    # img_scale in config is (w, h) in some dumps and (h, w) in others;
    # YOLOX uses square so both are equal.
    size = max(target_h, target_w)
    h0, w0 = image_bgr.shape[:2]
    scale = min(size / h0, size / w0)
    nh = int(round(h0 * scale))
    nw = int(round(w0 * scale))
    resized = cv.resize(image_bgr, (nw, nh), interpolation=cv.INTER_LINEAR)
    canvas = np.full((size, size, 3), pad_val, dtype=np.float32)
    # Pad bottom-right (MMDet Pad default).
    canvas[:nh, :nw] = resized.astype(np.float32)
    return canvas, float(scale), (h0, w0)


def _meshgrid_priors(
    feat_h: int,
    feat_w: int,
    stride: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """MlvlPointGenerator(offset=0) with_stride=True → [cx, cy, stride, stride]."""
    shift_y = torch.arange(feat_h, dtype=dtype, device=device) * stride
    shift_x = torch.arange(feat_w, dtype=dtype, device=device) * stride
    shift_yy, shift_xx = torch.meshgrid(shift_y, shift_x, indexing="ij")
    centers = torch.stack([shift_xx, shift_yy], dim=-1).reshape(-1, 2)
    strides = centers.new_full((centers.shape[0], 2), float(stride))
    return torch.cat([centers, strides], dim=-1)


def decode_yolox_outputs(
    cls_scores: list[torch.Tensor],
    bbox_preds: list[torch.Tensor],
    objectnesses: list[torch.Tensor],
    *,
    num_classes: int,
    strides: tuple[int, ...] = (8, 16, 32),
    score_thr: float,
    nms_iou: float,
    scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return xyxy (original image), scores, labels as numpy arrays."""
    num_imgs = cls_scores[0].shape[0]
    assert num_imgs == 1, "batch=1 inference only"

    flatten_cls = []
    flatten_bbox = []
    flatten_obj = []
    flatten_priors = []
    for cls_score, bbox_pred, objectness, stride in zip(
        cls_scores, bbox_preds, objectnesses, strides
    ):
        _, _, feat_h, feat_w = cls_score.shape
        priors = _meshgrid_priors(
            feat_h, feat_w, stride, cls_score.dtype, cls_score.device
        )
        flatten_priors.append(priors)
        flatten_cls.append(
            cls_score.permute(0, 2, 3, 1).reshape(1, -1, num_classes)
        )
        flatten_bbox.append(bbox_pred.permute(0, 2, 3, 1).reshape(1, -1, 4))
        flatten_obj.append(objectness.permute(0, 2, 3, 1).reshape(1, -1))

    cls_all = torch.cat(flatten_cls, dim=1).sigmoid()[0]
    bbox_all = torch.cat(flatten_bbox, dim=1)[0]
    obj_all = torch.cat(flatten_obj, dim=1).sigmoid()[0]
    priors = torch.cat(flatten_priors, dim=0)

    xys = bbox_all[:, :2] * priors[:, 2:] + priors[:, :2]
    whs = bbox_all[:, 2:].exp() * priors[:, 2:]
    tl = xys - whs / 2
    br = xys + whs / 2
    boxes = torch.cat([tl, br], dim=-1)
    if scale > 0:
        boxes = boxes / scale

    max_scores, labels = torch.max(cls_all, dim=1)
    scores = max_scores * obj_all
    keep = scores >= score_thr
    boxes = boxes[keep]
    scores = scores[keep]
    labels = labels[keep]
    if boxes.numel() == 0:
        empty = np.zeros((0, 4), dtype=np.float32)
        return empty, np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.int32)

    keep_idx = batched_nms(boxes, scores, labels, nms_iou)
    boxes = boxes[keep_idx]
    scores = scores[keep_idx]
    labels = labels[keep_idx]
    return (
        boxes.detach().cpu().numpy().astype(np.float32),
        scores.detach().cpu().numpy().astype(np.float32),
        labels.detach().cpu().numpy().astype(np.int32),
    )


class MmdetYoloxDetector:
    """Thread-safe-ish detector: call ``detect`` under the shared device lock."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        device: str | None = None,
        imgsz: int | None = None,
        score_thr: float | None = None,
        nms_iou: float | None = None,
        class_name: str = "vehicle",
    ) -> None:
        if device is None or str(device) == "":
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        # Ultralytics-style device strings: "0" / "cpu"
        if str(device).isdigit():
            device = f"cuda:{device}"
        self.device = torch.device(device)
        self.model, meta = build_yolox_from_checkpoint(checkpoint_path, self.device)
        self.meta = meta
        size = int(imgsz) if imgsz else int(meta.img_scale[0])
        self.img_scale = (size, size)
        self.score_thr = float(score_thr) if score_thr is not None else float(meta.score_thr)
        self.nms_iou = float(nms_iou) if nms_iou is not None else float(meta.nms_iou)
        self.num_classes = int(meta.num_classes)
        self.names = {i: class_name if self.num_classes == 1 else str(i) for i in range(self.num_classes)}
        if self.num_classes == 1:
            self.names[0] = class_name

    @torch.inference_mode()
    def detect(
        self,
        image_bgr: np.ndarray,
        *,
        conf: float | None = None,
        iou: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (xyxy, scores, class_ids) in original image coordinates."""
        canvas, scale, (h0, w0) = letterbox_square(image_bgr, self.img_scale)
        # MMDet YOLOX DefaultFormatBundle: BGR float CHW, values in 0–255.
        tensor = (
            torch.from_numpy(canvas)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(self.device, dtype=torch.float32)
        )
        cls_scores, bbox_preds, objectnesses = self.model(tensor)
        score_thr = float(conf) if conf is not None else self.score_thr
        nms_iou = float(iou) if iou is not None else self.nms_iou
        xyxy, scores, labels = decode_yolox_outputs(
            cls_scores,
            bbox_preds,
            objectnesses,
            num_classes=self.num_classes,
            score_thr=score_thr,
            nms_iou=nms_iou,
            scale=scale,
        )
        if xyxy.size:
            xyxy[:, [0, 2]] = xyxy[:, [0, 2]].clip(0, w0 - 1)
            xyxy[:, [1, 3]] = xyxy[:, [1, 3]].clip(0, h0 - 1)
        return xyxy, scores, labels
