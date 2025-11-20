import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
import os
from PIL import Image
from torchvision import transforms

# --------------------------
# 1. Core Transformer Components
# --------------------------

class TransformerBlock(nn.Module):
    """
    A single Vision Transformer Encoder block (Multi-Head Attention + MLP).
    Based on standard Transformer architecture.
    """
    def __init__(self, embed_dim, num_heads, mlp_dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # MLP Block: Expand -> GELU -> Contract
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # Multi-Head Attention (MHA)
        # x is B x N x D (Batch x Sequence Length x Embedding Dim)
        
        # 1. Norm and MHA
        x_norm = self.norm1(x)
        # attn_output is B x N x D, attn_weights are not needed here
        attn_output, _ = self.attn(x_norm, x_norm, x_norm) 
        x = x + attn_output  # Residual connection
        
        # 2. Norm and MLP
        x_norm = self.norm2(x)
        x = x + self.mlp(x_norm) # Residual connection
        return x

# --------------------------
# 2. Hybrid Feature Extractor
# --------------------------

class HybridImageExtractor(nn.Module):
    def __init__(self, 
                 cnn_out_channels=2048, # Output dimension of ResNet50
                 embed_dim=256,        # Dimension for Transformer embedding
                 num_transformer_blocks=4, 
                 num_heads=8, 
                 mlp_dim=512):
        
        super().__init__()
        self.embed_dim = embed_dim

        ## 1. CNN Backbone (Local Feature Extractor)
        # Use a pre-trained ResNet50 up to the average pooling layer
        # We use the 'DEFAULT' weights for ImageNet pre-training
        resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        # Use all layers except the final AdaptiveAvgPool2d and FC layer
        self.cnn_backbone = nn.Sequential(*list(resnet.children())[:-2])
        
        # Freeze CNN weights (optional, but recommended for medical imaging pre-training)
        # for param in self.cnn_backbone.parameters():
        #     param.requires_grad = False
        
        # The output feature map from ResNet50 (layer4) will have 2048 channels.
        # We need a linear projection to match the Transformer's embed_dim.
        self.projection = nn.Conv2d(cnn_out_channels, embed_dim, kernel_size=1)

        ## 2. Transformer Encoder (Global Context)
        # Positional Encoding is crucial for ViT/Transformers
        # Since the CNN feature map size depends on the input size, we'll use 
        # a simple Learnable Position Embedding and let the network learn to resize.
        # We will assume an input size that results in a standard 7x7 feature map (49 patches).
        # max_sequence_length = 7 * 7 + 1 (if using a class token)
        self.pos_embedding = nn.Parameter(torch.randn(1, 49, embed_dim) * 0.02)
        
        # Create a sequence of Transformer Blocks
        self.transformer_encoder = nn.Sequential(
            *[TransformerBlock(embed_dim, num_heads, mlp_dim) 
              for _ in range(num_transformer_blocks)]
        )

        ## 3. Global Average Pooling (GAP)
        # AdaptiveAvgPool2d(output_size=1) performs Global Average Pooling
        # This will convert the B x D x H x W feature map back to B x D x 1 x 1
        self.global_avg_pooling = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        # Input x shape: (B, 3, H, W)
        
        # --- CNN Backbone ---
        # Output shape: (B, 2048, H/32, W/32) e.g., (B, 2048, 7, 7) for 224x224 input
        cnn_features = self.cnn_backbone(x)
        
        # --- Projection ---
        # Project 2048 channels down to embed_dim (e.g., 256)
        # Output shape: (B, embed_dim, H', W') e.g., (B, 256, 7, 7)
        proj_features = self.projection(cnn_features)
        
        # --- Prepare for Transformer ---
        B, D, H, W = proj_features.shape
        # Flatten the spatial dimensions (H' * W') into a sequence length N
        # Output shape: (B, H'*W', D) e.g., (B, 49, 256)
        sequence = proj_features.flatten(2).transpose(1, 2)
        
        # Add Positional Embedding (assuming 7x7=49 tokens)
        # Note: For non-224x224 inputs, the positional embedding would need interpolation
        sequence = sequence + self.pos_embedding 
        
        # --- Transformer Encoder ---
        # Output shape remains: (B, 49, 256)
        transformer_output = self.transformer_encoder(sequence)

        # --- Global Average Pooling (GAP) ---
        # 1. Reverse flatten: turn (B, 49, 256) back into (B, 256, 7, 7)
        # Note: This is an alternative to pooling the sequence,
        # but often cleaner to pool the resulting feature map.
        # We reshape the sequence back to a feature map shape.
        feature_map = transformer_output.transpose(1, 2).view(B, D, H, W)

        # 2. Apply GAP
        # Output shape: (B, D, 1, 1) e.g., (B, 256, 1, 1)
        pooled_features = self.global_avg_pooling(feature_map)
        
        # 3. Squeeze to get the final feature vector
        # Output shape: (B, D) e.g., (B, 256)
        image_feature_vector = pooled_features.squeeze()
        
        return image_feature_vector

# --------------------------
# 3. Example Usage
# --------------------------

if __name__ == '__main__':
    # Initialize the model
    # The output feature vector will have a size equal to the embed_dim (256)
    model = HybridImageExtractor(embed_dim=256)
    print("--- Model Architecture ---")
    print(model)
    
    # --- Load and Preprocess Images from Folder ---
    image_folder = 'processed_images_training'
    image_files = [f for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    # Define image transformations to match model input
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load images and stack them into a batch
    image_tensors = []
    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)
        # Open image and ensure it's in RGB format
        image = Image.open(image_path).convert('RGB')
        image_tensors.append(preprocess(image))
    
    # Raise an error if no images are found
    if not image_tensors:
        raise ValueError(f"No images found in the '{image_folder}' directory.")

    # Stack individual image tensors into a single batch tensor
    image_batch = torch.stack(image_tensors)
    
    # Perform forward pass with the batch of real images
    print("\n--- Testing Forward Pass with Real Images ---")
    with torch.no_grad():
        output_features = model(image_batch)
    
    # Check output size
    print(f"Input Shape (B, C, H, W): {image_batch.shape}")
    print(f"Output Feature Vector Shape (B, D): {output_features.shape}")

    # Expected output shape: (number_of_images, 256)
    if output_features.shape == (len(image_files), 256):
        print("✅ Feature extraction successful and dimension is correct.")
    else:
        print("❌ Feature extraction failed or dimension is incorrect.")