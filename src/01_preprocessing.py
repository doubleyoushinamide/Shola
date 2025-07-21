import os
import pandas as pd
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# Define preprocessing transformations with augmentation
 

# Custom Dataset class for Sholy images
class SholaDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.image_paths = []
        self.captions = []
        
        # Traverse directory structure to collect image paths and generate captions
        for emotion in os.listdir(root_dir):
            emotion_dir = os.path.join(root_dir, emotion)
            if os.path.isdir(emotion_dir):
                for view in ['front_view', 'side_view']:
                    view_dir = os.path.join(emotion_dir, view)
                    if os.path.isdir(view_dir):
                        for img_file in os.listdir(view_dir):
                            img_path = os.path.join(view_dir, img_file)
                            caption = f"{view.replace('_', ' ')} of a {emotion} Sholy"
                            self.image_paths.append(img_path)
                            self.captions.append(caption)
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        caption = self.captions[idx]
        # Load with PIL, resize
        image = Image.open(img_path).convert("RGB")
        image = image.resize((512, 512))
        # Gentle augmentation: RandomHorizontalFlip
        if transforms.RandomHorizontalFlip(p=0.5)(Image.new('RGB', (1, 1))).getpixel((0, 0)) == (0, 0, 0):
            # This is a hack to get a random bool; the actual flip is applied below
            if transforms.RandomHorizontalFlip(p=0.5)(Image.new('RGB', (1, 1))).getpixel((0, 0)) == (0, 0, 0):
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
        # ToTensor and normalization
        image = transforms.ToTensor()(image)
        image = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])(image)
        return image, caption

# Main preprocessing function
def preprocess_and_save():
    # Input and output directories
    input_dir = '/kaggle/input/image-input-main/image_input'
    output_dir = '/kaggle/working/processed_images'
    # output_format = '/kaggle/input/processed_images'
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize dataset
    dataset = SholaDataset(input_dir)
    caption_data = []

    # Process and save images (no DataLoader, direct iteration)
    for i in range(len(dataset)):
        img, caption = dataset[i]
        # Save normalized tensor as PNG for inspection (unnormalize first)
        img_np = img.numpy().transpose(1, 2, 0)
        img_np = (img_np * 0.5) + 0.5  # Unnormalize
        img_np = (img_np * 255).clip(0, 255).astype('uint8')
        img_pil = Image.fromarray(img_np)
        filename = f'image_{i}.png'
        img_path = os.path.join(output_dir, filename)
        img_pil.save(img_path)
        caption_data.append({'image_path': filename, 'caption': caption})

    # Save captions to CSV (image_path is just the filename)
    caption_df = pd.DataFrame(caption_data)
    caption_df.to_csv('/kaggle/working/captions.csv', index=False)
    print(f"Processed {len(caption_data)} images and saved with captions to /kaggle/working/")

# Execute preprocessing
if __name__ == "__main__":
    preprocess_and_save()