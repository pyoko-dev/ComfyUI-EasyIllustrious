# IllustriousAutoOutpaint

Single-node, Illustrious-first outpainting for seamless image expansion and creative borders.

## What does it do?

- Expands images with soft-mask padding and noise, tuned for anime/illustration.
- Supports two-stage growth for big expansions and optional refiner pass.
- Scene hints for landscape, portrait, street, and more.
- Handles denoise_mask/noise_mask for clean borders.

## Inputs

- **image**: The image to outpaint.
- **mask**: Optional mask for where to expand.
- **prompt**: Conditioning prompt for the new area.
- **steps/cfg/sampler/scheduler/denoise**: Control the outpaint process.
- **scene_hint**: Add context (e.g., landscape, portrait).
- **feather_px**: Softens the border for seamless blending.
- **noise_outside**: Controls how much noise is added to the new area.
- **refiner**: Optional, for extra detail in the expanded area.

## Outputs

- **outpainted image**: The expanded image, ready for further processing or display.

## Artist Tips

- Use feathering for smooth transitions between old and new areas.
- Scene hints can help guide the AI for more natural expansions.
- Try two-stage growth for large outpaints to avoid artifacts.
- Combine with MultiPass or TriplePass for advanced workflows.

## Example Workflow

1. Start with a finished image.
2. Mask the area you want to expand (or let the node auto-mask borders).
3. Use AutoOutpaint with a creative prompt and scene hint.
4. Feed the result into a sampler for further refinement.

---
For more on creative expansion, see [IllustriousMultiPassSampler](IllustriousMultiPassSampler.md).
