import torch
from diffusers import StableDiffusionPipeline
from peft import LoraConfig, get_peft_model
import os

#This is the same as 03_finetuning.py
def load_lora_unet(pipe, checkpoint_path):
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["to_q", "to_k", "to_v", "to_out.0"]
    )
    unet = get_peft_model(pipe.unet, lora_config)
    checkpoint = torch.load(checkpoint_path, map_location="cuda")
    unet.load_state_dict(checkpoint['model_state_dict'])
    return unet

def generate_images(prompt, pipe, unet, num_images=3, height=512, width=512, guidance_scale=7.5, num_inference_steps=50, seed=None):
    pipe.unet = unet
    generator = torch.Generator(device="cuda")
    if seed is not None:
        generator.manual_seed(seed)
    images = []
    for i in range(num_images):
        result = pipe(
            prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator
        )
        images.append(result.images[0])
    return images

def main():
    checkpoint_path = '/kaggle/input/lora-finetuned/lora_finetuned.pth'  # Update path as needed
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16
    )
    pipe.safety_checker = None
    pipe.to("cuda")
    unet = load_lora_unet(pipe, checkpoint_path)
    while True:
        prompt = input("Enter your prompt (or 'exit' to quit): ")
        if prompt.lower() == 'exit':
            break
        images = generate_images(prompt, pipe, unet, num_images=3)
        for idx, img in enumerate(images):
            out_path = f"generated_image_{idx+1}.png"
            img.save(out_path)
            print(f"Saved: {out_path}")
    print("Inference complete.")

if __name__ == "__main__":
    main()
