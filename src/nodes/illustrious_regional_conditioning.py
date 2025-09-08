from __future__ import annotations

from typing import List, Tuple
import torch


class IllustriousRegionalConditioning:
    """
    Compose regional (mask-scoped) prompts into a CONDITIONING object.
    - Takes a base/global conditioning and appends region-specific entries with masks at latent resolution.
    - Compatible with samplers that support masked/area conditioning (extras include: mask, timestep_percent_range, weight).
    - Does not touch the latent; only modifies CONDITIONING.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP", {"tooltip": "CLIP used to encode regional prompts."}),
                "base_cond": ("CONDITIONING", {"tooltip": "Base/global conditioning to start from."}),
                "latent": ("LATENT", {"tooltip": "Latent (for sizing masks to H/8 x W/8)."}),
                "normalize_overlap": ("BOOLEAN", {"default": True, "tooltip": "Normalize overlapping masks so combined weights per pixel don't exceed 1."}),
                "mask_blur": ("INT", {"default": 4, "min": 0, "max": 64, "step": 1}),
                "mask_dilate": ("INT", {"default": 2, "min": 0, "max": 64, "step": 1}),
            },
            "optional": {
                # Up to 6 regions (expand as needed)
                "mask_1": ("MASK", {}), "prompt_1": ("STRING", {"multiline": True}),
                "weight_1": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 4.0, "step": 0.05}),
                "start_1": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "end_1":   ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),

                "mask_2": ("MASK", {}), "prompt_2": ("STRING", {"multiline": True}),
                "weight_2": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 4.0, "step": 0.05}),
                "start_2": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "end_2":   ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),

                "mask_3": ("MASK", {}), "prompt_3": ("STRING", {"multiline": True}),
                "weight_3": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 4.0, "step": 0.05}),
                "start_3": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "end_3":   ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),

                "mask_4": ("MASK", {}), "prompt_4": ("STRING", {"multiline": True}),
                "weight_4": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 4.0, "step": 0.05}),
                "start_4": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "end_4":   ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),

                "mask_5": ("MASK", {}), "prompt_5": ("STRING", {"multiline": True}),
                "weight_5": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 4.0, "step": 0.05}),
                "start_5": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "end_5":   ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),

                "mask_6": ("MASK", {}), "prompt_6": ("STRING", {"multiline": True}),
                "weight_6": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 4.0, "step": 0.05}),
                "start_6": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "end_6":   ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "STRING")
    RETURN_NAMES = ("conditioning", "info")
    FUNCTION = "build"
    CATEGORY = "Easy Illustrious / Conditioning"

    # ---- helpers ----
    def _refine_mask(self, m_bchw, blur_px=0, dilate_px=0):
        m = m_bchw
        if dilate_px and dilate_px > 0:
            k = max(1, int(dilate_px))
            pad = k // 2
            m = torch.nn.functional.max_pool2d(m, kernel_size=k, stride=1, padding=pad)
        if blur_px and blur_px > 0:
            k = max(1, int(blur_px) | 1)
            pad = k // 2
            w = torch.ones((1, 1, k, k), device=m.device, dtype=m.dtype) / float(k * k)
            m = torch.nn.functional.conv2d(m, w, padding=pad)
        return torch.clamp(m, 0.0, 1.0)

    def _mask_to_latent_res(self, mask, latent_hw, device):
        # Convert mask to [B,1,lh,lw]
        m = mask
        if not torch.is_tensor(m):
            m = torch.tensor(m)
        m = m.to(device=device, dtype=torch.float32)

        if m.dim() == 2:
            m = m.unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
        elif m.dim() == 3:
            if m.shape[-1] == 1:
                m = m.permute(2, 0, 1).unsqueeze(0)  # [1,1,H,W]
            else:
                m = m.unsqueeze(0)  # [1,H,W] -> [1,1,H,W] slice first channel
                if m.shape[1] != 1:
                    m = m[:, :1, :, :]
        elif m.dim() == 4:
            if m.shape[-1] == 1:
                m = m.permute(0, 3, 1, 2).contiguous()
            elif m.shape[1] != 1:
                m = m[:, :1, :, :]

        m = torch.nn.functional.interpolate(m, size=latent_hw, mode="bilinear", align_corners=False)
        return torch.clamp(m, 0.0, 1.0)

    def _collect_regions(self, kwargs):
        regions = []
        for i in range(1, 7):
            mk = kwargs.get(f"mask_{i}", None)
            pk = kwargs.get(f"prompt_{i}", None)
            if mk is None or pk is None or len(str(pk).strip()) == 0:
                continue
            regions.append((
                mk, str(pk),
                float(kwargs.get(f"weight_{i}", 1.0)),
                float(kwargs.get(f"start_{i}", 0.0)),
                float(kwargs.get(f"end_{i}", 1.0)),
            ))
        return regions

    # ---- main ----
    def build(self, clip, base_cond, latent,
              normalize_overlap=True, mask_blur=4, mask_dilate=2, **region_kwargs):

        device = latent["samples"].device
        _, _, lh, lw = latent["samples"].shape
        latent_hw = (lh, lw)

        regions = self._collect_regions(region_kwargs)
        cond = list(base_cond)  # shallow copy

        # Refine and collect masks
        refined_masks: List[torch.Tensor] = []
        for (mask, prompt, weight, start, end) in regions:
            m = self._mask_to_latent_res(mask, latent_hw, device)
            m = self._refine_mask(m, blur_px=mask_blur, dilate_px=mask_dilate)
            refined_masks.append(m)

        # Normalize overlaps if requested
        if normalize_overlap and len(refined_masks) > 0:
            stack = torch.stack(refined_masks, dim=0)  # [R,B,1,lh,lw]
            denom = torch.clamp(stack.sum(dim=0), min=1e-6)  # [B,1,lh,lw]
            refined_masks = [m / denom for m in refined_masks]

        info_lines: List[str] = []
        for idx, ((mask, prompt, weight, start, end), m_bchw) in enumerate(zip(regions, refined_masks), start=1):
            # Encode prompt using CLIP
            tokens = clip.tokenize(prompt)
            region_cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)

            extras = {
                "pooled_output": pooled,
                "mask": m_bchw,  # [B,1,lh,lw]
                "weight": float(weight),
                "timestep_percent_range": (float(start), float(end)),
            }
            cond.append([region_cond, extras])
            info_lines.append(f"R{idx}: w={weight:.2f}, t=[{start:.2f},{end:.2f}], prompt={prompt[:64]}{'...' if len(prompt)>64 else ''}")

        info = "Illustrious Regional Conditioning\n" + ("\n".join(info_lines) if info_lines else "No regions added.")
        return (cond, info)
