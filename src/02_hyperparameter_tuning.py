import optuna
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from diffusers import StableDiffusionPipeline
from peft import LoraConfig, get_peft_model
import pandas as pd
from PIL import Image
import os

# Define image transformations to match Stable Diffusion expectations
transform = transforms.Compose([
    transforms.ToTensor(),  # Converts to float32
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    transforms.Lambda(lambda x: x.to(dtype=torch.float16))  # Convert to float16
])

# Custom Dataset class to load Sholy images and captions
class SholyDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.valid_paths = []
        for idx in range(len(self.data)):
            img_filename = self.data.iloc[idx]['image_path']  # Now just the filename
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
        img_filename = self.data.iloc[data_idx]['image_path']  # Now just the filename
        img_path = os.path.join(self.img_dir, img_filename)
        caption = self.data.iloc[data_idx]['caption']
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, caption

# Objective function for Optuna to minimize training loss
def objective(trial):
    # Define hyperparameter search space
    lr = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_int('batch_size', 1, 4, step=1)  # Lower batch size for OOM safety
    
    # Load dataset from Kaggle input directory
    csv_file = '/kaggle/input/image-preprocessor/captions.csv'
    img_dir = '/kaggle/input/image-preprocessor/processed_images'
    dataset = SholyDataset(csv_file, img_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)  # num_workers=0 for less memory
    
    # Load Stable Diffusion model in half-precision for P100 GPU
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", 
        torch_dtype=torch.float16
    )
    pipe.to("cuda")
    
    # Apply LoRA to U-Net attention layers
    lora_config = LoraConfig(
        r=16, 
        lora_alpha=32, 
        target_modules=[
            "to_q",
            "to_k",
            "to_v",
            "to_out.0"
        ]
    )
    unet = get_peft_model(pipe.unet, lora_config)
    
    # Freeze text encoder and VAE
    pipe.text_encoder.requires_grad_(False)
    pipe.vae.requires_grad_(False)
    
    # Set up optimizer with LoRA parameters only
    optimizer = torch.optim.AdamW(unet.parameters(), lr=lr, weight_decay=1e-2)
    
    # Training loop for 5 epochs
    import gc
    num_epochs = 5
    total_loss = 0
    grad_accum_steps = 2  # Accumulate gradients over 2 batches (adjust as needed)
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        for i, batch in enumerate(dataloader):
            images, captions = batch
            images = images.to("cuda")  # Images are already float16 from transform

            # Tokenize captions for text encoder
            inputs = pipe.tokenizer(
                captions,
                padding="max_length",
                max_length=77,
                truncation=True,
                return_tensors="pt"
            ).to("cuda")
            text_embeddings = pipe.text_encoder(inputs.input_ids)[0]

            # Encode images to latent space
            latents = pipe.vae.encode(images).latent_dist.sample() * 0.18215  # Scale as per Stable Diffusion

            # Add noise and predict with U-Net
            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, 1000, (latents.shape[0],), device="cuda").long()
            noisy_latents = pipe.scheduler.add_noise(latents, noise, timesteps)
            predicted_noise = unet(noisy_latents, timesteps, text_embeddings).sample

            # Compute mean squared error loss
            loss = torch.nn.functional.mse_loss(predicted_noise, noise)
            loss = loss / grad_accum_steps  # Normalize loss for accumulation
            loss.backward()

            if (i + 1) % grad_accum_steps == 0 or (i + 1) == len(dataloader):
                optimizer.step()
                optimizer.zero_grad()

            # Free up memory after each batch
            torch.cuda.empty_cache()
            gc.collect()

            total_loss += loss.item() * grad_accum_steps  # Undo normalization for reporting

    # Return average loss over all batches and epochs
    avg_loss = total_loss / (num_epochs * len(dataloader))
    return avg_loss

# Run Optuna study with 10 trials
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=10)

# Display best trial results
print('Best trial:')
trial = study.best_trial
print(f'  Value: {trial.value}')
print(f'  Params:')
for key, value in trial.params.items():
    print(f'    {key}: {value}')

# Save best hyperparameters as JSON
import json
with open('/kaggle/working/best_hyperparameters.json', 'w') as f:
    json.dump(trial.params, f, indent=2)
print('Best hyperparameters saved to /kaggle/working/best_hyperparameters.json')