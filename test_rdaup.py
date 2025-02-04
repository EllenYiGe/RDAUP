import torch
import torch.nn.functional as F

from datasets.dataset_utils import get_office31_dataloaders
from models.backbone import get_resnet_backbone
from models.attention import CoordinateAttention
from models.rdaup_model import ClassifierHead, DomainDiscriminator
from utils.metrics import accuracy_score


def evaluate_rdaup(
    model_path,
    test_root,
    model_name="resnet50",
    class_num=31,
    batch_size=32,
    device=None
):
    """
    Load the trained RDAUP model and evaluate classification accuracy on labeled target domain data.
    test_root: Should have a similar subfolder structure to the source domain (one folder per class) to parse labels.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("===== Loading Test Data ... =====")
    # Get test data (for_test=True => test augmentation)
    _, test_loader = get_office31_dataloaders(
        source_root="",    # Not needed
        target_root=test_root,
        batch_size=batch_size,
        for_test=True
    )
    if test_loader is None:
        print("[Error] test_loader is empty, cannot test. Please check the data path.")
        return

    # Build model structure
    backbone, feat_dim = get_resnet_backbone(model_name, pretrained=False)
    coordinate_att = CoordinateAttention(in_channels=feat_dim, out_channels=feat_dim)
    classifier = ClassifierHead(in_feature_dim=feat_dim, class_num=class_num)
    domain_disc = DomainDiscriminator(in_feature_dim=feat_dim)  # If needed

    # Load weights
    print(f"===== Loading weights from '{model_path}' =====")
    ckpt = torch.load(model_path, map_location=device)
    backbone.load_state_dict(ckpt["backbone"])
    coordinate_att.load_state_dict(ckpt["coordinate_att"])
    classifier.load_state_dict(ckpt["classifier"])
    domain_disc.load_state_dict(ckpt["domain_disc"])

    backbone.to(device).eval()
    coordinate_att.to(device).eval()
    classifier.to(device).eval()

    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for imgs, labels in test_loader:
            # If the target domain label is -1, accuracy cannot be computed
            if labels[0] == -1:
                continue
            imgs, labels = imgs.to(device), labels.to(device)

            feats = backbone(imgs).view(imgs.size(0), -1)
            feats_2d = feats.view(feats.size(0), feats.size(1), 1, 1)
            feats_2d = coordinate_att(feats_2d)
            feats = feats_2d.view(feats.size(0), -1)

            logits = classifier(feats)
            preds = torch.argmax(logits, dim=1)
            correct = (preds == labels).sum().item()
            total_correct += correct
            total_samples += labels.size(0)

    if total_samples > 0:
        acc = total_correct / total_samples
        print(f"Accuracy on the test set ({total_samples} images): {acc * 100:.2f}%")
    else:
        print("No labels or no data to evaluate in the test set.")


if __name__ == "__main__":
    # Example
    evaluate_rdaup(
        model_path="rdaup_office31_final.pth",
        test_root="./data/Office31/webcam",
        model_name="resnet50",
        class_num=31,
        batch_size=32
    )
