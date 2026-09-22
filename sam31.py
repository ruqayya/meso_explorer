"""SAM 3.1 still-image inference using Meta's Python modules, without ComfyUI.

The small adapter retains the saved SAM3_Detect workflow's independent corner
and text branches. SAM's upstream revision is pinned because its component
builders are private. No video tracker, server, or workflow engine is loaded.
"""

from importlib.resources import files
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from sam3 import model_builder as builder
from sam3.model.box_ops import box_cxcywh_to_xyxy
from sam3.model.data_misc import FindStage
from sam3.model.necks import Sam3TriViTDetNeck
from sam3.model.vl_combiner import SAM3VLBackboneTri
from sam3.sam.mask_decoder import MaskDecoder
from sam3.sam.prompt_encoder import PromptEncoder
from sam3.sam.transformer import TwoWayTransformer

SAM_REVISION = "2345a4ad109ac29c569da749c91d84f10dc08c40"
DEFAULT_PROMPT = "Segment the human cell centered in this brightfield image"


def resize(tensor, shape, antialias=False):
    return F.interpolate(
        tensor.float(), size=shape, mode="bilinear", align_corners=False, antialias=antialias
    )


def build_transformer():
    # Same decoder as upstream, with lazy coordinate caching on the input device
    # instead of an unconditional CUDA allocation during construction.
    layer = builder.TransformerDecoderLayer(
        activation="relu",
        d_model=256,
        dim_feedforward=2048,
        dropout=0.1,
        cross_attention=builder.MultiheadAttention(num_heads=8, dropout=0.1, embed_dim=256, use_fa3=False),
        n_heads=8,
        use_text_cross_attention=True,
    )
    decoder = builder.TransformerDecoder(
        layer=layer,
        num_layers=6,
        num_queries=200,
        return_intermediate=True,
        box_refine=True,
        num_o2m_queries=0,
        dac=True,
        boxRPB="log",
        d_model=256,
        frozen=False,
        interaction_layer=None,
        dac_use_selfatt_ln=True,
        resolution=None,
        stride=14,
        use_act_checkpoint=True,
        presence_token=True,
    )
    return builder.TransformerWrapper(
        encoder=builder._create_transformer_encoder(), decoder=decoder, d_model=256
    )


class Sam31:
    def __init__(self, checkpoint, device="cuda", prompt=DEFAULT_PROMPT):
        self.device = torch.device(device)
        vocab = str(files("sam3").joinpath("assets/bpe_simple_vocab_16e6.txt.gz"))
        # Skip the upstream builder's unconditional CUDA position-cache allocation.
        visual = Sam3TriViTDetNeck(
            trunk=builder._create_vit_backbone(),
            position_encoding=builder._create_position_encoding(),
            d_model=256,
            scale_factors=[4.0, 2.0, 1.0],
        )
        backbone = SAM3VLBackboneTri(visual=visual, text=builder._create_text_encoder(vocab), scalp=0)
        self.detector = builder._create_sam3_model(
            backbone,
            build_transformer(),
            builder._create_geometry_encoder(),
            builder._create_segmentation_head(),
            builder._create_dot_product_scoring(),
            None,
            True,
        )
        self.prompt_encoder = PromptEncoder(
            embed_dim=256, image_embedding_size=(72, 72), input_image_size=(1008, 1008), mask_in_chans=16
        )
        self.decoder = MaskDecoder(
            transformer_dim=256,
            transformer=TwoWayTransformer(depth=2, embedding_dim=256, mlp_dim=2048, num_heads=8),
            num_multimask_outputs=3,
            use_high_res_features=True,
            pred_obj_scores=True,
            pred_obj_scores_mlp=True,
            use_multimask_token_for_obj_ptr=True,
        )
        path = Path(checkpoint)
        state = (
            load_file(str(path))
            if path.suffix == ".safetensors"
            else torch.load(path, map_location="cpu", weights_only=True)
        )
        if "model" in state:
            state = state["model"]
        # Official facebook/sam3.1 .pt and the workflow's fp16 safetensors share these keys.
        for module, prefix in [
            (self.detector, "detector."),
            (self.prompt_encoder, "tracker.model.interactive_sam_prompt_encoder."),
            (self.decoder, "tracker.model.interactive_sam_mask_decoder."),
        ]:
            weights = {k[len(prefix) :]: v for k, v in state.items() if k.startswith(prefix)}
            if not weights:
                raise ValueError(f"Not a SAM 3.1 multiplex checkpoint: missing {prefix}")
            # FP16 conversions omit deterministic complex RoPE buffers and the
            # unused pooled-text projection. All learned inference weights are required.
            if module is self.detector:
                for key, value in module.state_dict().items():
                    if key.endswith(".attn.freqs_cis") and key not in weights:
                        weights[key] = value
                projection = "backbone.language_backbone.encoder.text_projection"
                if projection not in weights:
                    weights[projection] = torch.zeros_like(module.state_dict()[projection])
            module.load_state_dict(weights, strict=True)
            module.eval().to(self.device)
        self.no_memory = (
            state["tracker.model.interactivity_no_mem_embed"].to(self.device).reshape(1, 256, 1, 1)
        )
        del state
        self.find_stage = FindStage(
            img_ids=torch.tensor([0], device=self.device),
            text_ids=torch.tensor([0], device=self.device),
            input_boxes=None,
            input_boxes_mask=None,
            input_boxes_label=None,
            input_points=None,
            input_points_mask=None,
        )
        with torch.inference_mode(), self.autocast():
            self.text = self.detector.backbone.forward_text([prompt], device=self.device)

    def autocast(self):
        # Upstream ViT's fused MLP emits bfloat16, including on CPU.
        return torch.autocast(self.device.type, dtype=torch.bfloat16)

    def encode(self, rgb, detect=True):
        tensor = (
            torch.from_numpy(np.array(rgb, dtype=np.float32)).permute(2, 0, 1)[None].to(self.device) / 255
        )
        # The source ComfyUI node passes [0,1] RGB directly to the backbone.
        tensor = resize(tensor, (1008, 1008))
        backbone = self.detector.backbone.forward_image(
            tensor, need_sam3_out=detect, need_interactive_out=True, need_propagation_out=False
        )
        features = [level.tensors for level in backbone["interactive"]["backbone_fpn"]]
        high = [self.decoder.conv_s0(features[0]), self.decoder.conv_s1(features[1])]
        image_embedding = features[-1] + self.no_memory
        return backbone, (image_embedding, high)

    def decode(self, embeddings, points=None, mask=None):
        if points is None:
            points = (
                torch.zeros(1, 1, 2, device=self.device),
                torch.full((1, 1), -1, dtype=torch.int32, device=self.device),
            )
        mask_prompt = resize(mask, (288, 288), antialias=True) if mask is not None else None
        sparse, dense = self.prompt_encoder(points=points, boxes=None, masks=mask_prompt)
        logits, _, _, object_score = self.decoder(
            image_embeddings=embeddings[0],
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            multimask_output=False,
            repeat_image=False,
            high_res_features=embeddings[1],
        )
        logits = torch.where(object_score[:, :, None, None] > 0, logits, -1024.0)
        return resize(logits, (1008, 1008))

    @torch.inference_mode()
    def predict(self, rgb, threshold=0.6, refine_iterations=2, corner_inset=4):
        with self.autocast():
            w, h = rgb.size
            backbone, embeddings = self.encode(rgb)
            x0, y0 = min(corner_inset, w - 1), min(corner_inset, h - 1)
            x1, y1 = max(0, w - corner_inset), max(0, h - corner_inset)
            coords = torch.tensor([[[x0, y0], [x0, y1], [x1, y0], [x1, y1]]], device=self.device).float()
            coords *= torch.tensor([1008 / w, 1008 / h], device=self.device)
            points = (coords, torch.zeros(1, 4, dtype=torch.int32, device=self.device))
            logits = self.decode(embeddings, points=points)
            for _ in range(max(0, refine_iterations - 1)):
                logits = self.decode(embeddings, mask=logits)
            corner_mask = resize(logits, (h, w))[0, 0] > 0
            backbone.update(self.text)
            output = self.detector.forward_grounding(
                backbone_out=backbone,
                find_input=self.find_stage,
                geometric_prompt=self.detector._get_dummy_prompt(),
                find_target=None,
            )
            # Source SAM3_Detect uses the class sigmoid, without presence multiplication.
            scores = output["pred_logits"][0, :, 0].sigmoid()
            best = int(scores.argmax())
            score = float(scores[best])
            count = int((scores > threshold).sum())
            combined = corner_mask
            if score > threshold:
                coarse = resize(output["pred_masks"][0, best][None, None], (h, w))
                combined = combined | (coarse[0, 0] > 0)
                if refine_iterations:
                    box = box_cxcywh_to_xyxy(output["pred_boxes"][0, best])
                    left, top, right, bottom = (box * box.new_tensor([w, h, w, h])).float().cpu().tolist()
                    bw, bh = right - left, bottom - top
                    left, top = max(0, int(left - 0.1 * bw)), max(0, int(top - 0.1 * bh))
                    right, bottom = min(w, int(right + 0.1 * bw)), min(h, int(bottom + 0.1 * bh))
                    if right > left and bottom > top:
                        _, crop_embeddings = self.encode(rgb.crop((left, top, right, bottom)), detect=False)
                        logits = coarse[:, :, top:bottom, left:right]
                        for _ in range(refine_iterations):
                            logits = self.decode(crop_embeddings, mask=resize(logits, (1008, 1008)))
                        refined = resize(logits, (bottom - top, right - left))[0, 0] > 0
                        combined[top:bottom, left:right] |= refined
            return combined.cpu().numpy(), {
                "text_score": score,
                "text_detections": count,
                "corner_area": int(corner_mask.sum()),
            }
