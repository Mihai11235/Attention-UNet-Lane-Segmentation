import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import accuracy_score, f1_score, jaccard_score
from tqdm import tqdm
import os
from PIL import Image

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


# Custom dataset class
class BDD100KDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.image_list = os.listdir(image_dir)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_list[idx])
        mask_path = os.path.join(self.mask_dir, self.image_list[idx].replace('.jpg', '.png'))

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert('L')

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask


# Evaluation function
def evaluate(model, val_loader, criterion):
    # model.eval()
    val_loss = 0.0
    val_accuracy = 0.0
    val_f1 = 0.0
    val_jaccard = 0.0
    num_batches = len(val_loader)

    with torch.no_grad():
        for images, masks in tqdm(val_loader, desc='Evaluating'):
            images = images.cuda()
            masks = masks.cuda()

            outputs = model(images)
            loss = criterion(outputs, masks)
            val_loss += loss.item()

            preds = torch.sigmoid(outputs).cpu().numpy() > 0.5
            true = masks.cpu().numpy() > 0.5

            val_accuracy += accuracy_score(true.flatten(), preds.flatten())
            val_f1 += f1_score(true.flatten(), preds.flatten())
            val_jaccard += jaccard_score(true.flatten(), preds.flatten())

    val_loss /= num_batches
    val_accuracy /= num_batches
    val_f1 /= num_batches
    val_jaccard /= num_batches

    print(f'Validation Loss: {val_loss:.4f}')
    print(f'Validation Accuracy: {val_accuracy:.4f}')
    print(f'Validation F1 Score: {val_f1:.4f}')
    print(f'Validation Jaccard Score: {val_jaccard:.4f}')


# Set the environment variable for CUDA
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.cuda.empty_cache()
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


# Define transformations for the images
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((256, 256))  # Resize to match the input size of the model
])


# Instantiate the model with attention
model_with_attention = UNetWithAttention(in_channels=3, out_channels=1).cuda()
model_with_attention.load_state_dict(torch.load('./best_unet_lane_detection.pth', weights_only=True)['model_state_dict'])
model_with_attention.eval()


# Load validation data
val_image_dir = './Dataset/100k_images_val_small_data/bdd100k/images/100k/val'
val_mask_dir = './Dataset/bdd100k_lane_labels_trainval_small_data/bdd100k/labels/lane/masks/val'
val_dataset = BDD100KDataset(val_image_dir, val_mask_dir, transform)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)


# Define loss function
criterion = nn.BCEWithLogitsLoss()
# Run evaluation
evaluate(model_with_attention, val_loader, criterion)