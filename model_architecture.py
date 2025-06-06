import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import ResNet50_Weights
import numpy as np

class AttentionModule(nn.Module):
    """Attention mechanism for feature enhancement"""
    
    def __init__(self, in_channels, reduction_ratio=16):
        super(AttentionModule, self).__init__()
        
        # Channel attention
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1),
            nn.Sigmoid()
        )
        
        # Spatial attention
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # Channel attention
        ca = self.channel_attention(x)
        x = x * ca
        
        # Spatial attention
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        spatial_input = torch.cat([avg_pool, max_pool], dim=1)
        sa = self.spatial_attention(spatial_input)
        x = x * sa
        
        return x

class FeaturePyramidNetwork(nn.Module):
    """Feature Pyramid Network for multi-scale feature extraction"""
    
    def __init__(self, in_channels_list, out_channels=256):
        super(FeaturePyramidNetwork, self).__init__()
        self.inner_blocks = nn.ModuleList()
        self.layer_blocks = nn.ModuleList()
        
        for in_channels in in_channels_list:
            self.inner_blocks.append(
                nn.Conv2d(in_channels, out_channels, 1)
            )
            self.layer_blocks.append(
                nn.Conv2d(out_channels, out_channels, 3, padding=1)
            )
    
    def forward(self, x_list):
        last_inner = self.inner_blocks[-1](x_list[-1])
        results = []
        results.append(self.layer_blocks[-1](last_inner))
        
        for idx in range(len(x_list) - 2, -1, -1):
            inner_lateral = self.inner_blocks[idx](x_list[idx])
            inner_top_down = F.interpolate(
                last_inner, size=inner_lateral.shape[-2:], 
                mode='bilinear', align_corners=False
            )
            last_inner = inner_lateral + inner_top_down
            results.insert(0, self.layer_blocks[idx](last_inner))
        
        return results

class ResNetBackbone(nn.Module):
    """Modified ResNet50 backbone for feature extraction"""
    
    def __init__(self, input_channels=1):
        super(ResNetBackbone, self).__init__()
        
        # Load pretrained ResNet50
        resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        
        # Modify first conv layer for single channel input
        if input_channels != 3:
            resnet.conv1 = nn.Conv2d(
                input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
        
        # Extract layers
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        
        self.layer1 = resnet.layer1  # 256 channels
        self.layer2 = resnet.layer2  # 512 channels
        self.layer3 = resnet.layer3  # 1024 channels
        self.layer4 = resnet.layer4  # 2048 channels
    
    def forward(self, x):
        features = []
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        features.append(x)  # C2
        
        x = self.layer2(x)
        features.append(x)  # C3
        
        x = self.layer3(x)
        features.append(x)  # C4
        
        x = self.layer4(x)
        features.append(x)  # C5
        
        return features

class LungCancerDetectionModel(nn.Module):
    """Complete lung cancer detection model with ResNet50 + FPN + Attention"""
    
    def __init__(self, num_classes=2, input_channels=1, dropout_rate=0.5):
        super(LungCancerDetectionModel, self).__init__()
        
        # Backbone
        self.backbone = ResNetBackbone(input_channels)
        
        # Feature Pyramid Network
        fpn_in_channels = [256, 512, 1024, 2048]
        self.fpn = FeaturePyramidNetwork(fpn_in_channels, out_channels=256)
        
        # Attention modules for each FPN level
        self.attention_modules = nn.ModuleList([
            AttentionModule(256) for _ in range(4)
        ])
        
        # Feature fusion
        self.feature_fusion = nn.Sequential(
            nn.Conv2d(256 * 4, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(64, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Extract backbone features
        backbone_features = self.backbone(x)
        
        # Apply FPN
        fpn_features = self.fpn(backbone_features)
        
        # Apply attention to each FPN level
        attended_features = []
        for i, (feature, attention) in enumerate(zip(fpn_features, self.attention_modules)):
            attended_feature = attention(feature)
            attended_features.append(attended_feature)
        
        # Resize all features to same spatial size
        target_size = attended_features[0].shape[-2:]
        resized_features = []
        
        for feature in attended_features:
            if feature.shape[-2:] != target_size:
                resized_feature = F.interpolate(
                    feature, size=target_size, mode='bilinear', align_corners=False
                )
            else:
                resized_feature = feature
            resized_features.append(resized_feature)
        
        # Concatenate features
        fused_features = torch.cat(resized_features, dim=1)
        
        # Feature fusion
        fused_features = self.feature_fusion(fused_features)
        
        # Global pooling
        pooled_features = self.global_pool(fused_features)
        pooled_features = pooled_features.view(pooled_features.size(0), -1)
        
        # Classification
        output = self.classifier(pooled_features)
        
        return output

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

def count_parameters(model):
    """Count total trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def model_summary(model, input_size=(1, 1, 512, 512)):
    """Print model summary"""
    device = next(model.parameters()).device
    x = torch.randn(input_size).to(device)
    
    print("Model Architecture:")
    print(model)
    print(f"\nTotal parameters: {count_parameters(model):,}")
    
    # Forward pass to get output shape
    model.eval()
    with torch.no_grad():
        output = model(x)
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {output.shape}")

class ResNetBackboneSmall(nn.Module):
    """Modified ResNet18 backbone for feature extraction - smaller than ResNet50"""
    
    def __init__(self, input_channels=1):
        super(ResNetBackboneSmall, self).__init__()
        
        # Load pretrained ResNet18
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        
        # Modify first conv layer for single channel input
        if input_channels != 3:
            resnet.conv1 = nn.Conv2d(
                input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
        
        # Extract layers
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        
        self.layer1 = resnet.layer1  # 64 channels
        self.layer2 = resnet.layer2  # 128 channels
        self.layer3 = resnet.layer3  # 256 channels
        self.layer4 = resnet.layer4  # 512 channels
    
    def forward(self, x):
        features = []
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        features.append(x)  # C2
        
        x = self.layer2(x)
        features.append(x)  # C3
        
        x = self.layer3(x)
        features.append(x)  # C4
        
        x = self.layer4(x)
        features.append(x)  # C5
        
        return features

class LungCancerDetectionModelSmall(nn.Module):
    """Smaller lung cancer detection model with ResNet18 + simplified FPN"""
    
    def __init__(self, num_classes=2, input_channels=1, dropout_rate=0.5):
        super(LungCancerDetectionModelSmall, self).__init__()
        
        # Backbone
        self.backbone = ResNetBackboneSmall(input_channels)
        
        # Feature Pyramid Network - simplified with fewer channels
        fpn_in_channels = [64, 128, 256, 512]  # ResNet18 channels
        self.fpn = FeaturePyramidNetwork(fpn_in_channels, out_channels=128)  # Reduced from 256 to 128
        
        # Attention modules for each FPN level - simplified
        self.attention_modules = nn.ModuleList([
            AttentionModule(128, reduction_ratio=8) for _ in range(4)  # Reduced channels and ratio
        ])
        
        # Feature fusion - simplified
        self.feature_fusion = nn.Sequential(
            nn.Conv2d(128 * 4, 256, 3, padding=1),  # Reduced input channels
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 3, padding=1),  # Reduced output channels
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Classifier - simplified
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),  # Reduced input features
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(64, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Extract backbone features
        backbone_features = self.backbone(x)
        
        # Apply FPN
        fpn_features = self.fpn(backbone_features)
        
        # Apply attention to each FPN level
        attended_features = []
        for i, (feature, attention) in enumerate(zip(fpn_features, self.attention_modules)):
            attended_feature = attention(feature)
            attended_features.append(attended_feature)
        
        # Resize all features to same spatial size
        target_size = attended_features[0].shape[-2:]
        resized_features = []
        
        for feature in attended_features:
            if feature.shape[-2:] != target_size:
                resized_feature = F.interpolate(
                    feature, size=target_size, mode='bilinear', align_corners=False
                )
            else:
                resized_feature = feature
            resized_features.append(resized_feature)
        
        # Concatenate features
        fused_features = torch.cat(resized_features, dim=1)
        
        # Feature fusion
        fused_features = self.feature_fusion(fused_features)
        
        # Global pooling
        pooled_features = self.global_pool(fused_features)
        pooled_features = pooled_features.view(pooled_features.size(0), -1)
        
        # Classification
        output = self.classifier(pooled_features)
        
        return output

# Example usage
if __name__ == "__main__":
    # Create model
    model = LungCancerDetectionModel(num_classes=2, input_channels=1)
    
    # Model summary
    model_summary(model)
    
    # Test forward pass
    x = torch.randn(2, 1, 512, 512)  # Batch of 2 images
    output = model(x)
    print(f"\nTest output shape: {output.shape}")
    print(f"Test output: {output}")
    
    # Test smaller model
    model_small = LungCancerDetectionModelSmall(num_classes=2, input_channels=1)
    print("\n--- Smaller Model Summary ---")
    model_summary(model_small)
    
    # Test focal loss
    criterion = FocalLoss(alpha=1, gamma=2)
    targets = torch.tensor([0, 1])
    loss = criterion(output, targets)
    print(f"Focal loss: {loss.item()}")