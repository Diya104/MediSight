import torch
import torch.nn as nn
import torch.nn.functional as F

class LabelCorrelationNetwork(nn.Module):
    """
    Adjusts the logits (pre-Sigmoid scores) of the 7 non-Diabetes diseases
    based on the predicted logit of Diabetes (index 0).
    """
    def __init__(self, num_diseases):
        super().__init__()
        self.num_diseases = num_diseases
        
        # 1. Prediction of the Correction Factor
        # The input is the logit for Diabetes (1 dimension)
        # The output is the learned adjustment factor for the other 7 diseases (num_diseases - 1)
        self.adjustment_predictor = nn.Sequential(
            nn.Linear(1, 16),
            nn.LeakyReLU(),
            nn.Linear(16, num_diseases - 1),
            # Use Tanh or Sigmoid to bound the adjustment factor
            nn.Tanh() # Bounding the adjustment to [-1, 1]
        )

        # 2. Learned Weights for Correlation
        # These weights determine HOW MUCH each disease is inherently correlated with Diabetes.
        # These 7 weights are learned during training.
        self.correlation_weights = nn.Parameter(torch.randn(1, num_diseases - 1))

    def forward(self, initial_logits):
        # Initial Logits shape: (B, 8)
        
        # --- 1. Extract Diabetes Logit ---
        # Assume Diabetes is the first disease (index 0)
        diabetes_logit = initial_logits[:, 0].unsqueeze(1) # Shape: (B, 1)
        
        # --- 2. Predict Adjustment Factor ---
        # Based on the Diabetes logit, predict a factor for the other 7 diseases
        # adjustment_factor shape: (B, 7)
        adjustment_factor = self.adjustment_predictor(diabetes_logit)
        
        # --- 3. Compute Final Correction ---
        # Correction = (Learned Correlation Weights) * (Predicted Adjustment Factor)
        # correction shape: (B, 7)
        correction = self.correlation_weights * adjustment_factor
        
        # --- 4. Apply Correction to Non-Diabetes Logits ---
        # Separate the logits for the 7 other diseases (indices 1 to 7)
        other_logits = initial_logits[:, 1:] # Shape: (B, 7)
        
        # Apply correction: Residual connection
        adjusted_other_logits = other_logits + correction
        
        # --- 5. Recombine Logits ---
        # The Diabetes logit itself is not adjusted by this correlation network
        adjusted_logits = torch.cat([diabetes_logit, adjusted_other_logits], dim=1) # Shape: (B, 8)
        
        return adjusted_logits
