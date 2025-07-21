import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from diffusers import StableDiffusionPipeline
from peft import LoraConfig, get_peft_model
import pandas as pd
from PIL import Image
import os
#Dataset loader same as 02_hyperparameter_tuning.py
def get_dataloader(csv_file, img_dir, batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        transforms.Lambda(lambda x: x.to(dtype=torch.float16))
    ])
    
    class SholyDataset(Dataset):
        def __init__(self, csv_file, img_dir, transform=None):
            self.data = pd.read_csv(csv_file)
            self.img_dir = img_dir
            self.transform = transform
            self.valid_paths = []
            for idx in range(len(self.data)):
                img_filename = self.data.iloc[idx]['image_path']
                img_path = os.path.join(self.img_dir, img_filename)
                if os.path.exists(img_path):
                    self.valid_paths.append(idx)
                else:
                    print(f"Warning: Image not found at {img_path}")
            if not self.valid_paths:
                raise ValueError("No valid image paths found in dataset")
        def __len__(self):
            return len(self.valid_paths)
        def __getitem__(self, idx):
            data_idx = self.valid_paths[idx]
            img_filename = self.data.iloc[data_idx]['image_path']
            img_path = os.path.join(self.img_dir, img_filename)
            caption = self.data.iloc[data_idx]['caption']
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, caption
    dataset = SholyDataset(csv_file, img_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    return dataloader

def main():
    csv_file = '/kaggle/input/image-preprocessor/captions.csv'
    img_dir = '/kaggle/input/image-preprocessor/processed_images'
    batch_size = 3
    lr = 2.528840890950286e-05
    num_epochs = 20
    grad_accum_steps = 2
    dataloader = get_dataloader(csv_file, img_dir, batch_size)
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16
    )
    pipe.to("cuda")
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["to_q", "to_k", "to_v", "to_out.0"]
    )
    unet = get_peft_model(pipe.unet, lora_config)
    pipe.text_encoder.requires_grad_(False)
    pipe.vae.requires_grad_(False)
    optimizer = torch.optim.AdamW(unet.parameters(), lr=lr, weight_decay=1e-2)
    import gc
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        total_loss = 0
        for i, batch in enumerate(dataloader):
            images, captions = batch
            images = images.to("cuda")
            inputs = pipe.tokenizer(
                captions,
                padding="max_length",
                max_length=77,
                truncation=True,
                return_tensors="pt"
            ).to("cuda")
            text_embeddings = pipe.text_encoder(inputs.input_ids)[0]
            latents = pipe.vae.encode(images).latent_dist.sample() * 0.18215
            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, 1000, (latents.shape[0],), device="cuda").long()
            noisy_latents = pipe.scheduler.add_noise(latents, noise, timesteps)
            predicted_noise = unet(noisy_latents, timesteps, text_embeddings).sample
            loss = torch.nn.functional.mse_loss(predicted_noise, noise)
            loss = loss / grad_accum_steps
            loss.backward()
            if (i + 1) % grad_accum_steps == 0 or (i + 1) == len(dataloader):
                optimizer.step()
                optimizer.zero_grad()
            torch.cuda.empty_cache()
            gc.collect()
            total_loss += loss.item() * grad_accum_steps
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {total_loss / len(dataloader):.6f}")
    # Save model
    torch.save({'model_state_dict': unet.state_dict()}, '/kaggle/working/lora_finetuned.pth')
    print("Model saved to /kaggle/working/lora_finetuned.pth")

if __name__ == "__main__":
    main()
