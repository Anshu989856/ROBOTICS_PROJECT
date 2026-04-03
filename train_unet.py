import torch
from torch.utils.data import Dataset, DataLoader
import os
import cv2
import numpy as np
from unet_model import UNet, TopologyPreservingLoss 

class CableDataset(Dataset):
    def __init__(self, img_dir, mask_dir):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        # Only grab the images we actually generated
        self.images = sorted([f for f in os.listdir(img_dir) if f.endswith('.jpg')])
        
    def __len__(self):
        return len(self.images)
        
    def __getitem__(self, idx):
        img_name = self.images[idx]
        mask_name = img_name.replace(".jpg", ".png")
        
        img_path = os.path.join(self.img_dir, img_name)
        mask_path = os.path.join(self.mask_dir, mask_name)
        
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # Resize to 256x256 for the U-Net architecture
        image = cv2.resize(image, (256, 256))
        mask = cv2.resize(mask, (256, 256))
        
        # Normalize and convert to PyTorch Tensors
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        mask = torch.from_numpy(mask).unsqueeze(0).float() / 255.0
        
        return image, mask

if __name__ == "__main__":
    # Setup Training Environment
    print("Initializing Phase 2 Training Pipeline...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using compute device: {device}")
    
    model = UNet(in_channels=3, out_channels=1).to(device)
    criterion = TopologyPreservingLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # Load the 400 augmented samples
    dataset = CableDataset("augmented_dataset/images", "augmented_dataset/masks")
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    print(f"Loaded {len(dataset)} samples. Starting 10 Epochs...")
    
    # The Training Loop
    for epoch in range(10): 
        model.train()
        epoch_loss = 0
        
        for imgs, masks in dataloader:
            imgs = imgs.to(device)
            masks = masks.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(imgs)
            
            # Calculate Topology-Aware Loss
            loss = criterion(outputs, masks)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/10] - Topology Loss: {avg_loss:.4f}")
    
    # Save the model weights
    torch.save(model.state_dict(), "cable_unet_midterm.pth")
    print("\nPhase 2 Complete! Model weights saved to 'cable_unet_midterm.pth'.")
