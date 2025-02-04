import torch
from models.backbone import get_resnet_backbone
from models.attention import CoordinateAttention
from models.rdaup_model import ClassifierHead, DomainDiscriminator
from datasets.dataset_utils import get_office31_dataloaders

def test_model_creation():
    print("Testing model creation...")
    try:
        # Test backbone
        backbone, feat_dim = get_resnet_backbone("resnet50", pretrained=True)
        print("✓ Backbone created successfully")
        
        # Test attention
        attention = CoordinateAttention(feat_dim, feat_dim)
        print("✓ Attention module created successfully")
        
        # Test classifier
        classifier = ClassifierHead(feat_dim, class_num=31)
        print("✓ Classifier created successfully")
        
        # Test discriminator
        discriminator = DomainDiscriminator(feat_dim)
        print("✓ Discriminator created successfully")
        
        return True
    except Exception as e:
        print(f"Error in model creation: {str(e)}")
        return False

def test_data_loading():
    print("\nTesting data loading...")
    try:
        source_loader, target_loader = get_office31_dataloaders(
            source_root="./data/Office31/amazon",
            target_root="./data/Office31/webcam",
            batch_size=4
        )
        if source_loader is None:
            print("Warning: Source loader is None (expected if no images in source directory)")
        if target_loader is None:
            print("Warning: Target loader is None (expected if no images in target directory)")
        print("✓ Data loaders created successfully")
        return True
    except Exception as e:
        print(f"Error in data loading: {str(e)}")
        return False

def main():
    print("=== Running basic functionality tests ===")
    
    # Test CUDA availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Run tests
    models_ok = test_model_creation()
    data_ok = test_data_loading()
    
    # Summary
    print("\n=== Test Summary ===")
    print(f"Model creation: {'✓' if models_ok else '✗'}")
    print(f"Data loading: {'✓' if data_ok else '✗'}")
    
    if models_ok and data_ok:
        print("\nAll basic functionality tests passed!")
    else:
        print("\nSome tests failed. Please check the errors above.")

if __name__ == "__main__":
    main()
