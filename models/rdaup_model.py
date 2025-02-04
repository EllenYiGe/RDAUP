import torch
import torch.nn as nn

class ClassifierHead(nn.Module):
    """
    Classifier head (two fully connected layers + BN + ReLU)
    """
    def __init__(self, in_feature_dim, class_num=31, hidden_dim=256):
        super(ClassifierHead, self).__init__()
        self.fc1 = nn.Linear(in_feature_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, class_num)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        logits = self.fc2(x)
        return logits


class DomainDiscriminator(nn.Module):
    """
    Domain discriminator (MLP), outputs 1D logits (for distinguishing domains via sigmoid)
    """
    def __init__(self, in_feature_dim, hidden_dim=1024):
        super(DomainDiscriminator, self).__init__()
        self.ad_layer = nn.Sequential(
            nn.Linear(in_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, f):
        return self.ad_layer(f)
