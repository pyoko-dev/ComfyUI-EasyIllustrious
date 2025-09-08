"""
Auto Outpaint (Illustrious) — Single-node, Illustrious-first outpainting.

Illustrious-tuned defaults (SDXL family), soft-mask padding + denoise_mask so only
borders are denoised, optional two-stage growth for big expansions, optional
two-phase schedule (grow→refine), optional refiner pass, and seam-toning micro-pass.
"""
from __future__ import annotations

import math
from typing import Tuple
import inspect

import torch
import torch.nn.functional as F

import comfy.model_management as mm
from nodes import (
    CLIPTextEncode,
    VAEDecode,
    VAEEncode,
    common_ksampler,
)

try:
    # Optional advanced encoder (BNK). Will be used if present.
    from nodes import BNK_CLIPTextEncodeAdvanced  # type: ignore
    _HAS_BNK_ADV = True
except Exception:
    BNK_CLIPTextEncodeAdvanced = None  # type: ignore
    _HAS_BNK_ADV = False


# Illustrious defaults
_ILLUSTRIOUS_DEFAULTS = dict(
    steps=30,
    cfg=5.5,
    sampler_name="euler_ancestral",
    scheduler="normal",
    denoise=0.65,
    noise_outside=1.0,
    feather_px=64,  # soft edge at image scale; applied at latent as //8
)

# Small, prompt-safe additive scene hints
_SCENE_HINTS = {
    "none": "",
    "landscape": ", wide landscape, horizon, depth, environmental storytelling",
    "interior": ", interior scene, natural light falloff, furniture context",
    "street": ", street scene, sidewalks, signage, city blocks, atmospheric haze",
    "portrait": ", environmental portrait, subject grounded in scene, lens bokeh",
}


def _pix_to_lat(p: int) -> int:
    # map image pixels → latent pixels (≈ ÷8); ceil to preserve requested area
    return (int(p) + 7) // 8


def _make_soft_border_mask_lat(
    h_lat: int, w_lat: int, pad_lat: Tuple[int, int, int, int], feather_lat: int, device: torch.device
) -> torch.Tensor:
    """
    Build a 1x1xH_latxW_lat mask with 1.0 in the outpaint border and 0.0 in the original area.
    Feather with a separable blur to avoid seams.
    """
    l, r, t, b = pad_lat
    m = torch.zeros(1, 1, h_lat, w_lat, device=device, dtype=torch.float32)
    if t > 0:
        m[:, :, :t, :] = 1.0
    if b > 0:
        m[:, :, h_lat - b :, :] = 1.0
    if l > 0:
        m[:, :, :, :l] = 1.0
    if r > 0:
        m[:, :, :, w_lat - r :] = 1.0

    if feather_lat > 0:
        # simple box blur approximator; small kernel at latent scale is enough
        k = max(1, 2 * (feather_lat // 2) + 1)  # odd
        if k > 1:
            # separable blur
            kernel_1d = torch.ones(1, 1, k, device=device, dtype=torch.float32) / k
            m = F.conv2d(m, kernel_1d.view(1, 1, k, 1), padding=(k // 2, 0))
            m = F.conv2d(m, kernel_1d.view(1, 1, 1, k), padding=(0, k // 2))
            m = m.clamp(0.0, 1.0)

    return m


def _pad_latent_with_center(
    latent_samples: torch.Tensor, pad_lat: Tuple[int, int, int, int]
) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """
    Pad latent on each side by (l,r,t,b) and return padded tensor + (y0, x0) index where the
    original latent top-left is placed inside the padded tensor.
    """
    l, r, t, b = pad_lat
    B, C, h, w = latent_samples.shape
    out = torch.zeros(B, C, h + t + b, w + l + r, device=latent_samples.device, dtype=latent_samples.dtype)
    y0, x0 = t, l
    out[:, :, y0 : y0 + h, x0 : x0 + w] = latent_samples
    return out, (y0, x0)


def _blend_keep_interior(
    new_latent: torch.Tensor, original_latent_padded: torch.Tensor, mask_border_soft: torch.Tensor
) -> torch.Tensor:
    """
    Blend to preserve interior (mask=0 keeps original; mask=1 uses new).
    """
    mask = mask_border_soft.to(dtype=new_latent.dtype, device=new_latent.device)
    return original_latent_padded * (1.0 - mask) + new_latent * mask


def _rgb_to_luma(img_nhwc: torch.Tensor) -> torch.Tensor:
    # img: [B,H,W,C] in [0,1]
    r, g, b = img_nhwc[..., 0], img_nhwc[..., 1], img_nhwc[..., 2]
    return (0.2126 * r + 0.7152 * g + 0.0722 * b).unsqueeze(-1)  # [B,H,W,1]


def _seam_tone(image_nhwc: torch.Tensor, mask_lat_b1: torch.Tensor, strength: float = 0.25, feather_px_img: int = 48) -> torch.Tensor:
    """
    Feathered luminance/color equalization along seam. `mask_lat_b1` is [B,1,H_lat,W_lat] with 1 at border.
    We upsample to image res and apply a gentle offset/blur only in the seam band.
    """
    if strength <= 0.0:
        return image_nhwc

    B, H, W, C = image_nhwc.shape
    # Upsample mask to image-space
    m_b1_img = F.interpolate(mask_lat_b1, size=(H, W), mode="bilinear", align_corners=False)  # [B,1,H,W]
    m_img = m_b1_img.permute(0, 2, 3, 1).contiguous()  # [B,H,W,1]

    # Build an interior-side band by inverting and blurring a bit
    inv = 1.0 - m_b1_img
    k = max(1, 2 * (feather_px_img // 2) + 1)
    if k > 1:
        # cheap separable blur to create soft bands
        k1 = torch.ones(1, 1, k, device=image_nhwc.device, dtype=image_nhwc.dtype) / k
        inv_blur = F.conv2d(inv, k1.view(1, 1, k, 1), padding=(k // 2, 0))
        inv_blur = F.conv2d(inv_blur, k1.view(1, 1, 1, k), padding=(0, k // 2))
        border_blur = F.conv2d(m_b1_img, k1.view(1, 1, k, 1), padding=(k // 2, 0))
        border_blur = F.conv2d(border_blur, k1.view(1, 1, 1, k), padding=(0, k // 2))
    else:
        inv_blur = inv
        border_blur = m_b1_img

    interior_band = (inv_blur - (inv_blur > 0).float() * 0.0).clamp(0, 1)  # [B,1,H,W]
    border_band = (border_blur - (border_blur > 0).float() * 0.0).clamp(0, 1) # [B,1,H,W]

    # Compute band-wise mean colors
    eps = 1e-6
    img_bchw = image_nhwc.permute(0, 3, 1, 2).contiguous()  # [B,C,H,W]
    i_sum = (img_bchw * interior_band).sum(dim=(2, 3), keepdim=True)
    b_sum = (img_bchw * border_band).sum(dim=(2, 3), keepdim=True)
    i_cnt = interior_band.sum(dim=(2, 3), keepdim=True).clamp_min(eps)
    b_cnt = border_band.sum(dim=(2, 3), keepdim=True).clamp_min(eps)
    mean_interior = i_sum / i_cnt
    mean_border = b_sum / b_cnt
    delta = (mean_interior - mean_border)  # [B,C,1,1]

    # Apply small color/luma offset only in the seam band
    blend_band = (border_band + interior_band).clamp(0.0, 1.0)  # [B,1,H,W]
    blend_nhwc = blend_band.permute(0, 2, 3, 1).contiguous()  # [B,H,W,1]
    corrected = image_nhwc + (delta.permute(0, 2, 3, 1) * blend_nhwc * strength)

    # Light local blur only on seam to hide ridge artifacts
    if k > 1:
        blur_src = corrected.permute(0, 3, 1, 2)
        blur = F.conv2d(blur_src, k1.view(1, 1, k, 1), padding=(k // 2, 0), groups=1)
        blur = F.conv2d(blur, k1.view(1, 1, 1, k), padding=(0, k // 2), groups=1)
        blur = blur.permute(0, 2, 3, 1)
        corrected = corrected * (1.0 - 0.15 * blend_nhwc) + blur * (0.15 * blend_nhwc)

    return corrected.clamp(0.0, 1.0)


class IllustriousAutoOutpaint:
    """
    Single node automatic outpainting for Illustrious (SDXL family).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "model": ("MODEL",),              # Base / UNet model (Illustrious)
                "clip": ("CLIP",),                # Dual-CLIP bundle for SDXL
                "vae": ("VAE",),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "negative": ("STRING", {"multiline": True, "default": ""}),
                # expansion (image pixels)
                "expand_left": ("INT", {"default": 256, "min": 0, "max": 4096, "step": 8}),
                "expand_right": ("INT", {"default": 256, "min": 0, "max": 4096, "step": 8}),
                "expand_top": ("INT", {"default": 256, "min": 0, "max": 4096, "step": 8}),
                "expand_bottom": ("INT", {"default": 256, "min": 0, "max": 4096, "step": 8}),
                # quality / guidance
                "steps": ("INT", {"default": _ILLUSTRIOUS_DEFAULTS["steps"], "min": 1, "max": 200}),
                "cfg": ("FLOAT", {"default": _ILLUSTRIOUS_DEFAULTS["cfg"], "min": 0.0, "max": 20.0}),
                "sampler_name": ([
                    "euler", "euler_ancestral", "ddim", "dpmpp_2m", "dpmpp_2m_sde", "dpmpp_sde", "heun", "lcm"
                ], {"default": _ILLUSTRIOUS_DEFAULTS["sampler_name"]}),
                "scheduler": (["normal", "karras", "exponential", "sgm_uniform"], {"default": _ILLUSTRIOUS_DEFAULTS["scheduler"]}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**31 - 1}),
                "denoise": ("FLOAT", {"default": _ILLUSTRIOUS_DEFAULTS["denoise"], "min": 0.0, "max": 1.0}),
                "noise_outside": ("FLOAT", {"default": _ILLUSTRIOUS_DEFAULTS["noise_outside"], "min": 0.0, "max": 1.0}),
                "feather_px": ("INT", {"default": _ILLUSTRIOUS_DEFAULTS["feather_px"], "min": 0, "max": 1024}),
                # nice extras
                "scene_hint": (list(_SCENE_HINTS.keys()), {"default": "none"}),
                "two_stage_growth": ("BOOLEAN", {"default": True}),  # half-pad then full
                # optional refiner pass
                "use_refiner": ("BOOLEAN", {"default": False}),
                # NEW: two-phase schedule (grow→refine)
                "two_phase_schedule": ("BOOLEAN", {"default": False}),
                "grow_steps_frac": ("FLOAT", {"default": 0.45, "min": 0.05, "max": 0.9, "step": 0.01}),
                "grow_denoise": ("FLOAT", {"default": 0.90, "min": 0.0, "max": 1.0, "step": 0.01}),
                "refine_denoise": ("FLOAT", {"default": 0.55, "min": 0.0, "max": 1.0, "step": 0.01}),
                # NEW: seam-toning micro-pass
                "seam_toning": ("BOOLEAN", {"default": True}),
                "seam_tone_strength": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01}),
                "seam_feather_px": ("INT", {"default": 48, "min": 8, "max": 256, "step": 4}),
            },
            "optional": {
                "refiner_model": ("MODEL",),     # if use_refiner=True
                "refiner_steps": ("INT", {"default": 12, "min": 1, "max": 60}),
                "refiner_denoise": ("FLOAT", {"default": 0.22, "min": 0.0, "max": 1.0}),
                # Advanced prompt weighting (uses BNK_CLIPTextEncodeAdvanced when available)
                "weight_interpretation": (
                    ["comfy", "A1111", "compel", "comfy++", "down_weight"],
                    {"default": "comfy", "tooltip": "How to apply token weights; requires BNK advanced encoder for non-comfy modes."},
                ),
                "token_normalization": (
                    ["none", "mean", "length", "length+mean"],
                    {"default": "none", "tooltip": "Normalize token weights; requires BNK advanced encoder to take effect."},
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "LATENT")
    RETURN_NAMES = ("image", "latent")
    FUNCTION = "run"
    CATEGORY = "Easy Illustrious / Composition"

    # ------------------- internal encode/sample/decode helpers ----------------

    def _encode_cond(self, clip, prompt: str, weight_interpretation: str | None = None, token_normalization: str | None = None):
        # Prefer BNK advanced encoder if available and non-default options requested
        try:
            use_adv = _HAS_BNK_ADV and (
                (weight_interpretation and weight_interpretation != "comfy") or (token_normalization and token_normalization != "none")
            )
            if use_adv and BNK_CLIPTextEncodeAdvanced is not None:
                enc = BNK_CLIPTextEncodeAdvanced()
                sig = inspect.signature(enc.encode)
                kwargs = {}
                if "weight_interpretation" in sig.parameters and weight_interpretation is not None:
                    kwargs["weight_interpretation"] = weight_interpretation
                if "token_normalization" in sig.parameters and token_normalization is not None:
                    kwargs["token_normalization"] = token_normalization
                out = enc.encode(clip, prompt, **kwargs)
                return out[0]
        except Exception as e:
            print(f"[AutoOutpaint] Advanced encoder unavailable or failed; falling back. Reason: {e}")
        # Fallback to default encoder
        return CLIPTextEncode().encode(clip, prompt)[0]

    def _ksampler_params(self):
        """Detect and cache supported parameters for common_ksampler across Comfy variants."""
        if not hasattr(self, "_ks_params"):
            try:
                self._ks_params = set(inspect.signature(common_ksampler).parameters.keys())
            except Exception:
                self._ks_params = set()
        return self._ks_params

    def _sample(
        self,
        model,
        positive,
        negative,
        latent_dict,
        seed: int,
        steps: int,
        cfg: float,
        sampler_name: str,
        scheduler: str,
        denoise: float,
        noise_mask: torch.Tensor | None = None,
        disable_noise: bool | None = None,
    ):
        # Call with positional args for broad compatibility; then add only supported kwargs
        args = (model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_dict)
        ks_params = self._ksampler_params()
        kwargs = {}
        kwargs["denoise"] = denoise
        if noise_mask is not None:
            if "denoise_mask" in ks_params:
                kwargs["denoise_mask"] = noise_mask
            elif "noise_mask" in ks_params:
                kwargs["noise_mask"] = noise_mask
        if disable_noise is not None and "disable_noise" in ks_params:
            kwargs["disable_noise"] = disable_noise
        samples = common_ksampler(*args, **kwargs)
        # Normalize sampler return into a LATENT dict with a tensor under "samples"
        tensor = None
        if isinstance(samples, torch.Tensor):
            tensor = samples
        elif isinstance(samples, dict) and "samples" in samples:
            maybe = samples["samples"]
            if isinstance(maybe, torch.Tensor):
                tensor = maybe
        elif isinstance(samples, (list, tuple)):
            for it in samples:
                if isinstance(it, torch.Tensor):
                    tensor = it
                    break
                if isinstance(it, dict) and "samples" in it and isinstance(it["samples"], torch.Tensor):
                    tensor = it["samples"]
                    break
        if tensor is None:
            try:
                tensor = getattr(samples, "samples")
            except Exception:
                tensor = None
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"Unsupported sampler return type: {type(samples)}")
        return {"samples": tensor}

    def _decode(self, vae, latent_dict):
        return VAEDecode().decode(vae, latent_dict)[0]

    def _encode_image_to_latent(self, vae, image):
        return VAEEncode().encode(vae, image)[0]

    # ----------------------------- main --------------------------------------

    def run(
        self,
        image,
        model,
        clip,
        vae,
        prompt,
        negative,
        expand_left,
        expand_right,
        expand_top,
        expand_bottom,
        steps,
        cfg,
        sampler_name,
        scheduler,
        seed,
        denoise,
        noise_outside,
        feather_px,
        scene_hint,
        two_stage_growth,
        use_refiner,
        two_phase_schedule,
        grow_steps_frac,
        grow_denoise,
        refine_denoise,
        seam_toning,
        seam_tone_strength,
        seam_feather_px,
        refiner_model=None,
        refiner_steps=12,
        refiner_denoise=0.22,
        weight_interpretation="comfy",
        token_normalization="none",
    ):
        device = mm.get_torch_device()

        # 0) augment prompt with scene hint
        if scene_hint in _SCENE_HINTS and _SCENE_HINTS[scene_hint]:
            prompt = (prompt or "") + _SCENE_HINTS[scene_hint]

        # 1) encode image → latent
        lat = self._encode_image_to_latent(vae, image)
        x0 = lat["samples"].to(device)

        # 2) compute latent-side padding and feather
        pad_lat = (
            _pix_to_lat(expand_left),
            _pix_to_lat(expand_right),
            _pix_to_lat(expand_top),
            _pix_to_lat(expand_bottom),
        )
        feather_lat = _pix_to_lat(feather_px)

        # Early exit if no expansion requested
        if sum(pad_lat) == 0:
            pos = self._encode_cond(clip, prompt, weight_interpretation, token_normalization)
            neg = self._encode_cond(clip, negative, weight_interpretation, token_normalization)
            lat_out = self._sample(model, pos, neg, {"samples": x0}, seed, steps, cfg, sampler_name, scheduler, denoise)
            img_out = self._decode(vae, lat_out)
            return img_out, lat_out

        # ------------------ inner pass (masked or unmasked) ------------------

        def one_outpaint_pass(
            x_in: torch.Tensor,
            pad: Tuple[int, int, int, int],
            steps_: int,
            denoise_: float,
            seed_: int,
            use_mask: bool,
        ):
            # Pad latent and build soft border mask
            x_padded, _ = _pad_latent_with_center(x_in, pad)
            Bp, Cp, H, W = x_padded.shape

            mask_border_soft = _make_soft_border_mask_lat(H, W, pad, feather_lat, device=x_padded.device)  # [1,1,H,W]
            mask_border_soft = mask_border_soft.to(dtype=x_padded.dtype)

            # Allow denoise/noise only in border
            noise_scale = float(noise_outside) if isinstance(noise_outside, (int, float)) else 1.0
            noise_mask_scaled = (mask_border_soft * max(0.0, min(1.0, noise_scale))).clamp(0.0, 1.0)  # [1,1,H,W]
            denoise_mask_b1 = noise_mask_scaled.expand(Bp, 1, H, W).contiguous()

            # Build conditionings
            pos = self._encode_cond(clip, prompt, weight_interpretation, token_normalization)
            neg = self._encode_cond(clip, negative, weight_interpretation, token_normalization)

            # Deterministic border noise
            g = torch.Generator(device=x_padded.device)
            g.manual_seed(int(seed_))
            border_noise = torch.randn_like(x_padded, generator=g)

            # Seed: if masked pass → seed only border; if unmasked → seed full latent
            if use_mask:
                x_start = x_padded * (1.0 - denoise_mask_b1) + border_noise * denoise_mask_b1
            else:
                x_start = border_noise  # full latent noise for free growth

            ks_params = self._ksampler_params()
            pass_mask_as_kw = ("denoise_mask" in ks_params) or ("noise_mask" in ks_params)
            lat_padded_dict = {"samples": x_start}

            sampled = self._sample(
                model,
                pos,
                neg,
                lat_padded_dict,
                seed_,
                steps_,
                cfg,
                sampler_name,
                scheduler,
                denoise_,
                noise_mask=(denoise_mask_b1 if (use_mask and pass_mask_as_kw) else None),
                disable_noise=None,
            )

            s = sampled["samples"]
            # Re-blend to guarantee interior identity if this was a masked pass
            if use_mask:
                if s.device != x_padded.device or s.dtype != x_padded.dtype:
                    s = s.to(device=x_padded.device, dtype=x_padded.dtype)
                s = _blend_keep_interior(s, x_padded, mask_border_soft)
            return {"samples": s}, denoise_mask_b1, mask_border_soft

        # ---------------- two-stage & two-phase orchestration ----------------

        # two-stage: grow half then full (reduces stretching/context shock)
        def staged_pad_sequence(pad):
            if two_stage_growth and any(p > _pix_to_lat(384) for p in pad):
                half = tuple(max(1, p // 2) for p in pad)
                rem = tuple(p - h for p, h in zip(pad, half))
                return [half, rem]
            return [pad]

        lat_current = {"samples": x0}
        last_mask_b1 = None
        last_soft_mask = None

        for idx, pad_chunk in enumerate(staged_pad_sequence(pad_lat), start=1):
            if two_phase_schedule:
                # Phase 1: unmasked grow
                grow_steps = max(8, int(steps * float(grow_steps_frac)))
                grow_dnz = float(max(denoise, grow_denoise))
                lat_grow, _, _ = one_outpaint_pass(
                    lat_current["samples"], pad_chunk, grow_steps, grow_dnz, seed, use_mask=False
                )
                # Phase 2: masked refine
                refine_steps = max(6, steps - grow_steps)
                ref_dnz = float(min(denoise, refine_denoise))
                lat_refine, mask_b1, soft_mask = one_outpaint_pass(
                    lat_grow["samples"], (0, 0, 0, 0), refine_steps, ref_dnz, seed, use_mask=True
                )
                lat_current = lat_refine
                last_mask_b1, last_soft_mask = mask_b1, soft_mask
            else:
                # Single masked pass for this stage
                eff_dnz = denoise
                if max(pad_chunk) >= _pix_to_lat(192):
                    eff_dnz = max(denoise, 0.90)
                lat_current, mask_b1, soft_mask = one_outpaint_pass(
                    lat_current["samples"], pad_chunk, steps, eff_dnz, seed, use_mask=True
                )
                last_mask_b1, last_soft_mask = mask_b1, soft_mask

        lat_out = lat_current

        # Optional refiner pass (lower denoise, separate model)
        if use_refiner and refiner_model is not None and refiner_steps > 0 and refiner_denoise > 0.0:
            pos_r = self._encode_cond(clip, prompt, weight_interpretation, token_normalization)
            neg_r = self._encode_cond(clip, negative, weight_interpretation, token_normalization)
            lat_out = self._sample(
                refiner_model,
                pos_r,
                neg_r,
                lat_out,
                seed,
                refiner_steps,
                cfg=max(4.8, min(cfg, 7.0)),
                sampler_name=sampler_name,
                scheduler=scheduler,
                denoise=refiner_denoise,
                noise_mask=None,
            )

        # Decode
        img_out = self._decode(vae, lat_out)

        # Seam-toning micro-pass (NHWC)
        if seam_toning and last_mask_b1 is not None:
            img_out = _seam_tone(
                img_out, last_mask_b1, strength=float(seam_tone_strength), feather_px_img=int(seam_feather_px)
            )

        return img_out, lat_out
