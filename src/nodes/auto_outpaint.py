"""
Auto Outpaint (Illustrious) — Single-node, Illustrious-first outpainting.

Illustrious-tuned defaults (SDXL family), soft-mask padding + noise_mask so only
borders are denoised, optional two-stage growth for big expansions, optional
refiner pass, and small scene bias hints.
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
    # Ensure mask matches tensor dtype/device for safe math
    mask = mask_border_soft.to(dtype=new_latent.dtype, device=new_latent.device)
    return original_latent_padded * (1.0 - mask) + new_latent * mask


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
                "two_stage_growth": ("BOOLEAN", {"default": True}),  # good for big pads
                # optional refiner pass
                "use_refiner": ("BOOLEAN", {"default": False}),
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
        # denoise is widely supported; pass it
        kwargs["denoise"] = denoise
        # Mask only if supported (prefer 'denoise_mask' per Comfy's KSampler)
        if noise_mask is not None:
            if "denoise_mask" in ks_params:
                kwargs["denoise_mask"] = noise_mask
            elif "noise_mask" in ks_params:
                kwargs["noise_mask"] = noise_mask
        # disable_noise only if supported
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
            # Prefer first tensor; otherwise look for dict with "samples"
            for it in samples:
                if isinstance(it, torch.Tensor):
                    tensor = it
                    break
                if isinstance(it, dict) and "samples" in it and isinstance(it["samples"], torch.Tensor):
                    tensor = it["samples"]
                    break
        if tensor is None:
            # Last resort: attribute access
            try:
                tensor = getattr(samples, "samples")
            except Exception:
                tensor = None
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"Unsupported sampler return type: {type(samples)}")
        return {"samples": tensor}

    def _decode(self, vae, latent_dict):
        # VAEDecode returns a tuple; first item is IMAGE
        return VAEDecode().decode(vae, latent_dict)[0]

    def _encode_image_to_latent(self, vae, image):
        # VAEEncode returns a tuple; first item is LATENT dict {"samples": tensor}
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
        refiner_model=None,
        refiner_steps=12,
        refiner_denoise=0.22,
    weight_interpretation="comfy",
    token_normalization="none",
    ):
        device = mm.get_torch_device()

        # 0) augment prompt with scene hint (tiny, non-invasive)
        if scene_hint in _SCENE_HINTS and _SCENE_HINTS[scene_hint]:
            prompt = (prompt or "") + _SCENE_HINTS[scene_hint]

        # 1) encode image → latent
        lat = self._encode_image_to_latent(vae, image)
        x0 = lat["samples"].to(device)
        B, C, h_lat, w_lat = x0.shape

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
            # No change: still allow a tiny denoise if >0 (acts like gentle img2img)
            lat_out = self._sample(
                model, pos, neg, {"samples": x0}, seed, steps, cfg, sampler_name, scheduler, denoise
            )
            img_out = self._decode(vae, lat_out)
            return img_out, lat_out

        def one_outpaint_pass(
            x_in: torch.Tensor, pad: Tuple[int, int, int, int], steps_: int, denoise_: float, seed_: int
        ):
            # 3) pad latent and build a soft border mask
            x_padded, (y0, x0_) = _pad_latent_with_center(x_in, pad)
            _, _, H, W = x_padded.shape

            mask_border_soft = _make_soft_border_mask_lat(H, W, pad, feather_lat, device=device)
            # noise mask: where to add noise (only in border area), scaled by noise_outside
            if isinstance(noise_outside, (int, float)):
                noise_scale = float(noise_outside)
            else:
                noise_scale = 1.0
            noise_mask_scaled = (mask_border_soft * max(0.0, min(1.0, noise_scale))).clamp(0.0, 1.0)
            # Expand to sampler-expected shape [B,C,H,W]
            Bc, Cc = x_padded.shape[0], x_padded.shape[1]
            mask_for_sampler = noise_mask_scaled
            if mask_for_sampler.shape[0] != Bc or mask_for_sampler.shape[1] != Cc:
                mask_for_sampler = mask_for_sampler.expand(Bc, Cc, H, W)

            # 4) build conditionings
            pos = self._encode_cond(clip, prompt, weight_interpretation, token_normalization)
            neg = self._encode_cond(clip, negative, weight_interpretation, token_normalization)

            # Increase denoise a bit for larger expansions to encourage content growth
            eff_denoise = denoise_
            if max(pad) >= _pix_to_lat(192):
                eff_denoise = max(denoise_, 0.90)

            # 5) sample with noise only in border, keep interior stable
            ks_params = self._ksampler_params()
            if ("denoise_mask" in ks_params) or ("noise_mask" in ks_params):
                # Seed explicit noise in the border, then restrict denoising to that region
                noise = torch.randn_like(x_padded)
                m = noise_mask_scaled.to(dtype=x_padded.dtype, device=x_padded.device)
                x_start = x_padded * (1.0 - m) + noise * m
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
                    eff_denoise,
                    noise_mask=mask_for_sampler.to(dtype=x_padded.dtype, device=x_padded.device),
                    disable_noise=None,
                )
            elif "disable_noise" in ks_params:
                # Emulate: inject explicit noise in the border and disable sampler noise
                noise = torch.randn_like(x_padded)
                m = noise_mask_scaled.to(dtype=x_padded.dtype, device=x_padded.device)
                x_start = x_padded * (1.0 - m) + noise * m
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
                    eff_denoise,
                    noise_mask=None,
                    disable_noise=True,
                )
            else:
                # Fallback: at least seed noise in the border, keep sampler noise enabled
                noise = torch.randn_like(x_padded)
                m = noise_mask_scaled.to(dtype=x_padded.dtype, device=x_padded.device)
                x_start = x_padded * (1.0 - m) + noise * m
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
                    eff_denoise,
                    noise_mask=None,
                    disable_noise=None,
                )

            # 6) optional fallback: if border is still near-flat (solid color), try a stronger unmasked grow
            with torch.no_grad():
                m1 = noise_mask_scaled.to(dtype=s.dtype, device=s.device)
                # compute variance only over border region
                border_vals = s * m1
                flat_score = border_vals.var().item() if border_vals.numel() > 0 else 0.0
            if flat_score < 1e-6:
                # Re-run from the same noise-seeded latent but without mask and with higher denoise
                stronger = self._sample(
                    model,
                    pos,
                    neg,
                    {"samples": x_start},
                    seed_,
                    steps_,
                    cfg,
                    sampler_name,
                    scheduler,
                    denoise=min(1.0, eff_denoise + 0.1),
                    noise_mask=None,
                    disable_noise=None,
                )
                s = stronger["samples"].to(device=x_padded.device, dtype=x_padded.dtype)

            # 7) re-blend to guarantee interior stability (align dtype/device)
            s = sampled["samples"]
            if s.device != x_padded.device or s.dtype != x_padded.dtype:
                s = s.to(device=x_padded.device, dtype=x_padded.dtype)
            x_new = _blend_keep_interior(s, x_padded, mask_border_soft)

            return {"samples": x_new}

        # two-stage: grow half then full for large pads (reduces stretching / context shock)
        if two_stage_growth and any(p > _pix_to_lat(384) for p in pad_lat):
            half_pad = tuple(max(1, p // 2) for p in pad_lat)
            # pass 1: shorter steps, milder denoise
            lat_mid = one_outpaint_pass(
                x0, half_pad, max(12, int(steps * 0.6)), max(0.35, denoise * 0.85), seed
            )  # keep seed stable
            # pass 2: the remaining pad (delta)
            rem_pad = tuple(p - hp for p, hp in zip(pad_lat, half_pad))
            lat_full = one_outpaint_pass(
                lat_mid["samples"], rem_pad, steps, denoise, seed
            )
            lat_out = lat_full
        else:
            # single pass
            lat_out = one_outpaint_pass(x0, pad_lat, steps, denoise, seed)

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
                cfg=max(4.8, min(cfg, 7.0)),  # modest CFG on refiner
                sampler_name=sampler_name,
                scheduler=scheduler,
                denoise=refiner_denoise,
                noise_mask=None,  # full latent refine
            )

        # 7) decode
        img_out = self._decode(vae, lat_out)
        return img_out, lat_out
