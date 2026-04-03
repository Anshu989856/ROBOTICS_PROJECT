import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# 1. Topology-Aware Loss Function Setup
# ==========================================
class TopologyPreservingLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super(TopologyPreservingLoss, self).__init__()
        self.alpha = alpha
        
    def forward(self, pred, target):
        # Base implementation using Soft Dice (Proxy for clDice scaffold)
        smooth = 1.0
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        intersection = (pred_flat * target_flat).sum()
        dice_loss = 1 - ((2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth))
        
        # Binary Cross Entropy for pixel-wise accuracy
        bce_loss = F.binary_cross_entropy(pred_flat, target_flat)
        
        # Combined loss
        return (self.alpha * dice_loss) + ((1 - self.alpha) * bce_loss)

# ==========================================
# 2. U-Net Architecture (For Thin Structures)
# ==========================================
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(UNet, self).__init__()
        
        # Encoder (Downsampling)
        self.down1 = DoubleConv(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bot = DoubleConv(128, 256)
        
        # Decoder (Upsampling)
        self.up1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(256, 128) # 128 + 128 (skip) = 256
        
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(128, 64)
        
        # Output layer
        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Encoder passes
        d1 = self.down1(x)
        d2 = self.down2(self.pool1(d1))
        
        # Bottleneck
        b = self.bot(self.pool2(d2))
        
        # Decoder passes with skip connections
        u1 = self.up1(b)
        u1 = torch.cat([d2, u1], dim=1) # Skip connection
        u1 = self.conv1(u1)
        
        u2 = self.up2(u1)
        u2 = torch.cat([d1, u2], dim=1) # Skip connection
        u2 = self.conv2(u2)
        
        return self.sigmoid(self.out_conv(u2))

# ==========================================
# 3. Architecture Validation Test
# ==========================================
if __name__ == "__main__":
    print("Initializing Topology-Aware U-Net Architecture...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Instantiate model and loss
    model = UNet(in_channels=3, out_channels=1).to(device)
    criterion = TopologyPreservingLoss().to(device)
    
    # Create a dummy image tensor (Batch Size=1, Channels=3, Height=256, Width=256)
    dummy_input = torch.randn(1, 3, 256, 256).to(device)
    dummy_target = torch.rand(1, 1, 256, 256).to(device) # Fake ground truth mask
    
    # Forward pass
    print(f"Input Shape: {dummy_input.shape}")
    output = model(dummy_input)
    print(f"Output Shape: {output.shape} (Ready for mask generation)")
    
    # Compute loss
    loss = criterion(output, dummy_target)
    print(f"Initial Topology Loss computed successfully: {loss.item():.4f}")
    print("\nPhase 2 Architecture Validation: COMPLETE.")
