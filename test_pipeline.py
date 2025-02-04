import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np

from models.backbone import get_resnet_backbone
from models.attention import CoordinateAttention
from models.rdaup_model import ClassifierHead, DomainDiscriminator
from datasets.dataset_utils import get_office31_dataloaders
from utils.metrics import accuracy
from utils.visualization import plot_attention_map

def create_dummy_images():
    """Create dummy images for testing"""
    print("Creating dummy test images...")
    domains = ['amazon', 'webcam']
    categories = ['backpack', 'bike', 'calculator']
    
    for domain in domains:
        for category in categories:
            path = f"data/Office31/{domain}/{category}"
            os.makedirs(path, exist_ok=True)
            
            # Create 5 dummy images per category
            for i in range(5):
                # Create a random RGB image
                img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
                img = Image.fromarray(img)
                img.save(f"{path}/image_{i}.jpg")
    
    print("✓ Created dummy test images")

def test_training_pipeline():
    print("\nTesting full training pipeline...")
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Model components
        backbone, feat_dim = get_resnet_backbone("resnet50", pretrained=True)
        attention = CoordinateAttention(feat_dim, feat_dim)
        classifier = ClassifierHead(feat_dim, class_num=3)  # 3 classes for our test
        discriminator = DomainDiscriminator(feat_dim)
        
        # Move models to device
        backbone = backbone.to(device)
        attention = attention.to(device)
        classifier = classifier.to(device)
        discriminator = discriminator.to(device)
        
        # Optimizers
        optim_backbone = optim.Adam(backbone.parameters(), lr=0.001)
        optim_attention = optim.Adam(attention.parameters(), lr=0.001)
        optim_classifier = optim.Adam(classifier.parameters(), lr=0.001)
        optim_discriminator = optim.Adam(discriminator.parameters(), lr=0.001)
        
        # Get data loaders
        source_loader, target_loader = get_office31_dataloaders(
            source_root="./data/Office31/amazon",
            target_root="./data/Office31/webcam",
            batch_size=2
        )
        
        # Mini training loop
        print("Running mini training loop...")
        for epoch in range(2):
            for batch_idx, ((source_data, source_label), (target_data, _)) in enumerate(zip(source_loader, target_loader)):
                if batch_idx >= 2:  # Only test with 2 batches
                    break
                
                # Move data to device
                source_data = source_data.to(device)
                source_label = source_label.to(device)
                target_data = target_data.to(device)
                
                # Forward pass through backbone
                source_feature = backbone(source_data)  # Shape: [B, C, 1, 1]
                target_feature = backbone(target_data)  # Shape: [B, C, 1, 1]
                
                # Apply attention
                source_attention = attention(source_feature)  # Shape: [B, C, 1, 1]
                target_attention = attention(target_feature)  # Shape: [B, C, 1, 1]
                
                # Flatten features for classifier and discriminator
                source_feature_flat = source_attention.view(source_attention.size(0), -1)  # Shape: [B, C]
                target_feature_flat = target_attention.view(target_attention.size(0), -1)  # Shape: [B, C]
                
                # Forward pass through classifier and discriminator
                source_output = classifier(source_feature_flat)
                domain_output_source = discriminator(source_feature_flat.detach())
                domain_output_target = discriminator(target_feature_flat.detach())
                
                # Calculate losses
                cls_loss = nn.CrossEntropyLoss()(source_output, source_label)
                domain_loss = nn.BCEWithLogitsLoss()(
                    domain_output_source, torch.ones_like(domain_output_source)
                ) + nn.BCEWithLogitsLoss()(
                    domain_output_target, torch.zeros_like(domain_output_target)
                )
                
                # Backward pass
                cls_loss.backward()
                domain_loss.backward()
                
                # Optimize
                optim_backbone.step()
                optim_attention.step()
                optim_classifier.step()
                optim_discriminator.step()
                
                # Zero gradients
                optim_backbone.zero_grad()
                optim_attention.zero_grad()
                optim_classifier.zero_grad()
                optim_discriminator.zero_grad()
                
                print(f"Epoch {epoch}, Batch {batch_idx}: cls_loss={cls_loss.item():.4f}, domain_loss={domain_loss.item():.4f}")
        
        print("✓ Training pipeline test completed successfully")
        return True
    except Exception as e:
        print(f"Error in training pipeline: {str(e)}")
        return False

def main():
    print("=== Running full pipeline test ===")
    
    # Create test data
    create_dummy_images()
    
    # Test training pipeline
    pipeline_ok = test_training_pipeline()
    
    # Summary
    print("\n=== Test Summary ===")
    print(f"Training pipeline: {'✓' if pipeline_ok else '✗'}")
    
    if pipeline_ok:
        print("\nAll pipeline tests passed!")
    else:
        print("\nSome tests failed. Please check the errors above.")

if __name__ == "__main__":
    main()
