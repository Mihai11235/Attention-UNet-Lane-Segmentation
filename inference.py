from PIL import Image
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os

plt.style.use('dark_background')


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


# Modify the UNet architecture to include Attention Gates

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


# Load the trained model
model_with_attention = UNetWithAttention(in_channels=3, out_channels=1).cuda()
model_with_attention.load_state_dict(
    torch.load('./best_unet_lane_detection.pth', weights_only=True)['model_state_dict'])
model_with_attention.eval()

# Transformations
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize to match the input size used during training
    transforms.ToTensor()
])


# Function to preprocess the image and mask
def preprocess(image_path, mask_path):
    image = Image.open(image_path).convert('RGB')
    mask = Image.open(mask_path).convert('L')  # Grayscale for mask
    original_size = image.size  # Keep original size
    image = transform(image)
    mask = transform(mask)
    return image, mask, original_size


# Function to post-process the output
def postprocess(output, original_size):
    output = output.squeeze().cpu().numpy()
    output = np.uint8(output * 255)  # Scale to 0-255
    output = Image.fromarray(output).resize(original_size, Image.BILINEAR)
    output = np.array(output)
    return output


# Function to display the results
def display_results(image_path, mask_path, predicted_mask):
    original_image = Image.open(image_path).convert('RGB')
    ground_truth_mask = Image.open(mask_path).convert('L')

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(original_image)
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title("Ground Truth Mask")
    plt.imshow(ground_truth_mask, cmap="gray")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title("Predicted Mask")
    plt.imshow(predicted_mask, cmap="gray")
    plt.axis('off')

    plt.show()


def run_inference(image_path, mask_path):
    image, mask, original_size = preprocess(image_path, mask_path)
    image = image.unsqueeze(0).cuda()

    with torch.no_grad():
        output = model_with_attention(image)
        output = torch.sigmoid(output)  # Apply sigmoid to get probabilities

    predicted_mask = postprocess(output, original_size)

    # Save predicted mask
    save_folder = './Dataset/Predicted'
    os.makedirs(save_folder, exist_ok=True)
    predicted_mask_image = Image.fromarray(predicted_mask)
    predicted_mask_image.save(os.path.join(save_folder, os.path.basename(mask_path)))

    display_results(image_path, mask_path, predicted_mask)


# Example usage
image_path = './Dataset/100k_images_val_small_data/bdd100k/images/100k/val/b1e0c01d-dd9e6e2f.jpg'
mask_path = './Dataset/bdd100k_lane_labels_trainval_small_data/bdd100k/labels/lane/masks/val/b1e0c01d-dd9e6e2f.png'
run_inference(image_path, mask_path)