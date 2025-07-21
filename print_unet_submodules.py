import torch
from diffusers import StableDiffusionPipeline

# Load Stable Diffusion pipeline (use float16 for P100 GPU)
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
)
pipe.to("cuda")

# Print all submodules of the U-Net with their names and types
def print_unet_submodules(unet, prefix="unet"):
    for name, module in unet.named_modules():
        print(f"{prefix}.{name}: {type(module)}")

print_unet_submodules(pipe.unet)
