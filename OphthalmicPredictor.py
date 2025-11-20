import torch.nn as nn

class OphthalmicPredictor(nn.Module):
    def __init__(self, num_diseases=8, feature_dim=256, age_feature_dim=128):
        super().__init__()
        
        # --- 1. Sub-Modules ---
        
        # A. Image Pathway (The Hybrid Component)
        # Assumes HybridImageExtractor outputs a vector of size feature_dim (e.g., 256)
        self.image_extractor = HybridImageExtractor(embed_dim=feature_dim) 
        
        # B. Clinical Pathway (Age Extractor)
        self.age_extractor = nn.Sequential(
            nn.Linear(1, age_feature_dim // 2),
            nn.ReLU(),
            nn.Linear(age_feature_dim // 2, age_feature_dim)
        )
        
        # --- 2. Fusion and Prediction ---
        
        # C. Fusion Layer: Input dimension is sum of image and age features
        fused_dim = feature_dim + age_feature_dim
        
        # D. Initial Prediction Head (Step 4 in previous table)
        self.initial_head = nn.Linear(fused_dim, num_diseases)
        
        # E. Label Correlation Network (LCN) (Step 5)
        # It takes 8 logits as input and outputs 8 adjusted logits.
        self.lcn = LabelCorrelationNetwork(num_diseases) # Define this class separately
        
        # F. Final Output (Step 6)
        self.sigmoid = nn.Sigmoid() 

    def forward(self, image, age):
        # 1. Feature Extraction
        image_features = self.image_extractor(image)  # Output: (B, 256)
        age_features = self.age_extractor(age.float().unsqueeze(1)) # Output: (B, 128)
        
        # 2. Feature Fusion (Concatenation)
        fused_features = torch.cat([image_features, age_features], dim=1) # Output: (B, 384)
        
        # 3. Initial Prediction Head
        initial_logits = self.initial_head(fused_features) # Output: (B, 8)
        
        # 4. Label Correlation Network (LCN)
        adjusted_logits = self.lcn(initial_logits) # Output: (B, 8)
        
        # 5. Final Probabilities
        probabilities = self.sigmoid(adjusted_logits) # Output: (B, 8)
        
        return probabilities