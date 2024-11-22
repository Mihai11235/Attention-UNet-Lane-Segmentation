import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import jaccard_score
from tqdm import tqdm


class BDD100KDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.images = os.listdir(images_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.images_dir, img_name)
        mask_path = os.path.join(self.masks_dir, img_name.replace('.jpg', '.png'))

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask


class AttentionGate(nn.Module):
    def __init__(self, in_channels, gating_channels, inter_channels):
        super(AttentionGate, self).__init__()

        self.W_g = nn.Conv2d(gating_channels, inter_channels, kernel_size=1, stride=1, padding=0)
        self.W_x = nn.Conv2d(in_channels, inter_channels, kernel_size=1, stride=1, padding=0)
        self.psi = nn.Conv2d(inter_channels, 1, kernel_size=1, stride=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, g):
        # g: gating signal from the decoder
        # x: feature map from the encoder
        g = self.W_g(g)
        x = self.W_x(x)

        # Combine the two signals
        attn = self.relu(g + x)
        attn = self.psi(attn)
        attn = self.sigmoid(attn)

        # Apply the attention map to the feature map x
        return x * attn


class UNetWithAttention(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNetWithAttention, self).__init__()

        def CBR(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

        self.enc1 = CBR(in_channels, 64)
        self.enc2 = CBR(64, 128)
        self.enc3 = CBR(128, 256)
        self.enc4 = CBR(256, 512)

        self.pool = nn.MaxPool2d(2)

        self.bottleneck = CBR(512, 1024)

        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.attn4 = AttentionGate(512, 512, 256)  # Attention Gate for the skip connection
        self.dec4 = CBR(768, 512)

        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.attn3 = AttentionGate(256, 256, 128)  # Attention Gate for the skip connection
        self.dec3 = CBR(384, 256)

        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.attn2 = AttentionGate(128, 128, 64)  # Attention Gate for the skip connection
        self.dec2 = CBR(192, 128)

        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.attn1 = AttentionGate(64, 64, 32)  # Attention Gate for the skip connection
        self.dec1 = CBR(96, 64)

        self.conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoding path
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))

        # Bottleneck
        bottleneck = self.bottleneck(self.pool(enc4))

        # Decoding path
        # dec4
        dec4 = self.upconv4(bottleneck)
        attn4 = self.attn4.forward(enc4, dec4)  # Apply attention gate
        dec4 = torch.cat((attn4, dec4), dim=1)  # Concatenate attention output and upsampled feature map
        dec4 = self.dec4(dec4)  # Adjust the number of input channels expected here

        # dec3
        dec3 = self.upconv3(dec4)
        attn3 = self.attn3.forward(enc3, dec3)  # Apply attention gate
        dec3 = torch.cat((attn3, dec3), dim=1)
        dec3 = self.dec3(dec3)

        # dec2
        dec2 = self.upconv2(dec3)
        attn2 = self.attn2.forward(enc2, dec2)  # Apply attention gate
        dec2 = torch.cat((attn2, dec2), dim=1)
        dec2 = self.dec2(dec2)

        # dec1
        dec1 = self.upconv1(dec2)
        attn1 = self.attn1.forward(enc1, dec1)  # Apply attention gate
        dec1 = torch.cat((attn1, dec1), dim=1)
        dec1 = self.dec1(dec1)

        return torch.sigmoid(self.conv(dec1))


def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=25, model_path='unet_lane_detection.pth', start_epoch=0):
    best_val_loss = float('inf')

    for epoch in range(start_epoch, num_epochs):
        model.train()
        train_loss = 0
        train_loader_tqdm = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} - Training')
        for images, masks in train_loader_tqdm:
            images = images.cuda()
            masks = masks.cuda()

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            train_loader_tqdm.set_postfix({'Loss': train_loss / len(train_loader.dataset)})

        train_loss = train_loss / len(train_loader.dataset)

        model.eval()
        val_loss = 0
        val_loader_tqdm = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} - Validation')
        with torch.no_grad():
            for images, masks in val_loader_tqdm:
                images = images.cuda()
                masks = masks.cuda()

                outputs = model(images)
                loss = criterion(outputs, masks)

                val_loss += loss.item() * images.size(0)
                val_loader_tqdm.set_postfix({'Loss': val_loss / len(val_loader.dataset)})

        val_loss = val_loss / len(val_loader.dataset)

        print(f'Epoch {epoch+1}/{num_epochs}, '
              f'Train Loss: {train_loss:.4f}, '
              f'Val Loss: {val_loss:.4f}')

        # Save the model after every 20 epoch
        if(epoch+1) % 20 == 0:
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch
            }, f'./attention/unet_lane_detection_epoch_{epoch+1}.pth')

        # Additionally, save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch
            }, './attention/best_unet_lane_detection.pth')



# Set the environment variable for CUDA
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])
train_dataset = BDD100KDataset(
    images_dir='./Dataset/100k_images_train_small_data/bdd100k/images/100k/train',
    masks_dir='./Dataset/bdd100k_lane_labels_trainval_small_data/bdd100k/labels/lane/masks/train',
    transform=transform
)
val_dataset = BDD100KDataset(
    images_dir='./Dataset/100k_images_val_small_data/bdd100k/images/100k/val',
    masks_dir='./Dataset/bdd100k_lane_labels_trainval_small_data/bdd100k/labels/lane/masks/val',
    transform=transform
)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)


# Instantiate the model with attention
model_with_attention = UNetWithAttention(in_channels=3, out_channels=1).cuda()

# Criterion and optimizer setup
criterion = nn.BCELoss()
optimizer = optim.Adam(model_with_attention.parameters(), lr=1e-3)

# Start training
train_model(model_with_attention, train_loader, val_loader, criterion, optimizer, num_epochs=20, model_path='unet_lane_detection.pth')
