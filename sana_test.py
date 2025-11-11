import torch
from diffusers import SanaPipeline, SanaVideoPipeline, DPMSolverMultistepScheduler
from diffusers import AutoencoderKLWan
from diffusers.utils import export_to_video


model_id = "Efficient-Large-Model/SANA-Video_2B_480p_diffusers"
pipe = SanaVideoPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
# pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=8.0)
pipe.vae.to(torch.float32)
pipe.text_encoder.to(torch.bfloat16)
pipe.to("cuda")
model_score = 30

prompt = "Evening, backlight, side lighting, soft light, high contrast, mid-shot, centered composition, clean solo shot. A tall man walks across a city"
negative_prompt = "A chaotic sequence with misshapen, deformed limbs in heavy motion blur, sudden disappearance, jump cuts, jerky movements, rapid shot changes, frames out of sync, inconsistent character shapes, temporal artifacts, jitter, and ghosting effects, creating a disorienting visual experience."
motion_prompt = f" motion score: {model_score}."
prompt = prompt + motion_prompt

video = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    height=256,
    width=256,
    frames=81,
    guidance_scale=6,
    num_inference_steps=100,
    generator=torch.Generator(device="cuda").manual_seed(42),
).frames[0]

export_to_video(video, "sana_video2.mp4", fps=16)
