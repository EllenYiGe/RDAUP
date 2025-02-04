import os
import time
import datetime
import logging
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import wandb

from models.backbone import get_resnet_backbone
from models.attention import CoordinateAttention
from models.rdaup_model import ClassifierHead, DomainDiscriminator
from datasets.dataset_utils import get_office31_dataloaders
from utils.metrics import accuracy
from utils.visualization import plot_attention_map

def setup_logger(log_dir):
    """Set up the logger"""
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    log_file = os.path.join(log_dir, f'training_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger()

def train(args, logger):
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Log experiment configuration
    logger.info("=== Experiment Configuration ===")
    for arg in vars(args):
        logger.info(f"{arg}: {getattr(args, arg)}")
    
    # Initialize model components
    backbone, feat_dim = get_resnet_backbone(args.model_name, pretrained=True, freeze_until=args.freeze_until)
    attention = CoordinateAttention(feat_dim, feat_dim)
    classifier = ClassifierHead(feat_dim, class_num=args.num_classes)
    discriminator = DomainDiscriminator(feat_dim)
    
    # Move models to device
    backbone = backbone.to(device)
    attention = attention.to(device)
    classifier = classifier.to(device)
    discriminator = discriminator.to(device)
    
    # Optimizers
    optimizer_backbone = optim.Adam(backbone.parameters(), lr=args.lr)
    optimizer_attention = optim.Adam(attention.parameters(), lr=args.lr)
    optimizer_classifier = optim.Adam(classifier.parameters(), lr=args.lr)
    optimizer_discriminator = optim.Adam(discriminator.parameters(), lr=args.lr)
    
    # Learning rate schedulers
    scheduler_backbone = optim.lr_scheduler.CosineAnnealingLR(optimizer_backbone, T_max=args.epochs)
    scheduler_attention = optim.lr_scheduler.CosineAnnealingLR(optimizer_attention, T_max=args.epochs)
    scheduler_classifier = optim.lr_scheduler.CosineAnnealingLR(optimizer_classifier, T_max=args.epochs)
    scheduler_discriminator = optim.lr_scheduler.CosineAnnealingLR(optimizer_discriminator, T_max=args.epochs)
    
    # Loss functions
    cls_criterion = nn.CrossEntropyLoss()
    domain_criterion = nn.BCEWithLogitsLoss()
    
    # Mixed precision training
    scaler = GradScaler()
    
    # Get data loaders
    source_loader, target_loader = get_office31_dataloaders(
        source_root=args.source_root,
        target_root=args.target_root,
        batch_size=args.batch_size
    )
    
    logger.info(f"Source dataset size: {len(source_loader.dataset)}")
    logger.info(f"Target dataset size: {len(target_loader.dataset)}")
    
    # Initialize wandb
    if args.use_wandb:
        wandb.init(
            project="RDAUP",
            name=f"{args.source_domain}2{args.target_domain}_{args.model_name}",
            config=args
        )
    
    # Training loop
    best_acc = 0.0
    total_time = 0.0
    
    logger.info("=== Starting Training ===")
    for epoch in range(args.epochs):
        epoch_start_time = time.time()
        
        # Set models to training mode
        backbone.train()
        attention.train()
        classifier.train()
        discriminator.train()
        
        # Training metrics
        train_cls_loss = 0.0
        train_domain_loss = 0.0
        train_total_loss = 0.0
        train_source_acc = 0.0
        train_domain_acc = 0.0
        num_batches = 0
        
        for batch_idx, ((source_data, source_label), (target_data, _)) in enumerate(zip(source_loader, target_loader)):
            # Move data to device
            source_data = source_data.to(device)
            source_label = source_label.to(device)
            target_data = target_data.to(device)
            
            # Mixed precision training
            with autocast():
                # Forward pass
                source_feature = backbone(source_data)
                target_feature = backbone(target_data)
                
                source_attention = attention(source_feature)
                target_attention = attention(target_feature)
                
                source_feature_flat = source_attention.view(source_attention.size(0), -1)
                target_feature_flat = target_attention.view(target_attention.size(0), -1)
                
                source_output = classifier(source_feature_flat)
                domain_output_source = discriminator(source_feature_flat.detach())
                domain_output_target = discriminator(target_feature_flat.detach())
                
                # Compute losses
                cls_loss = cls_criterion(source_output, source_label)
                domain_loss = domain_criterion(
                    domain_output_source, torch.ones_like(domain_output_source)
                ) + domain_criterion(
                    domain_output_target, torch.zeros_like(domain_output_target)
                )
                
                total_loss = cls_loss + args.lambda_adv * domain_loss
            
            # Backward pass
            scaler.scale(total_loss).backward()
            
            # Update parameters
            scaler.step(optimizer_backbone)
            scaler.step(optimizer_attention)
            scaler.step(optimizer_classifier)
            scaler.step(optimizer_discriminator)
            
            scaler.update()
            
            # Zero gradients
            optimizer_backbone.zero_grad()
            optimizer_attention.zero_grad()
            optimizer_classifier.zero_grad()
            optimizer_discriminator.zero_grad()
            
            # Update metrics
            train_cls_loss += cls_loss.item()
            train_domain_loss += domain_loss.item()
            train_total_loss += total_loss.item()
            train_source_acc += accuracy(source_output, source_label)
            train_domain_acc += 0.5 * (
                accuracy(domain_output_source > 0, torch.ones_like(domain_output_source, dtype=torch.long)) +
                accuracy(domain_output_target > 0, torch.zeros_like(domain_output_target, dtype=torch.long))
            )
            num_batches += 1
            
            # Print batch progress
            if (batch_idx + 1) % args.log_interval == 0:
                logger.info(
                    f"Epoch [{epoch+1}/{args.epochs}] Batch [{batch_idx+1}/{len(source_loader)}] "
                    f"Loss: {total_loss.item():.4f} "
                    f"(Cls: {cls_loss.item():.4f}, Domain: {domain_loss.item():.4f}) "
                    f"Acc: {accuracy(source_output, source_label):.2f}%"
                )
        
        # Calculate epoch average metrics
        avg_cls_loss = train_cls_loss / num_batches
        avg_domain_loss = train_domain_loss / num_batches
        avg_total_loss = train_total_loss / num_batches
        avg_source_acc = train_source_acc / num_batches
        avg_domain_acc = train_domain_acc / num_batches
        
        # Update learning rate
        scheduler_backbone.step()
        scheduler_attention.step()
        scheduler_classifier.step()
        scheduler_discriminator.step()
        
        # Calculate epoch time
        epoch_time = time.time() - epoch_start_time
        total_time += epoch_time
        
        # Log training results
        logger.info(
            f"Epoch [{epoch+1}/{args.epochs}] Time: {epoch_time:.2f}s | "
            f"Loss: {avg_total_loss:.4f} (Cls: {avg_cls_loss:.4f}, Domain: {avg_domain_loss:.4f}) | "
            f"Source Acc: {avg_source_acc:.2f}% | Domain Acc: {avg_domain_acc:.2f}% | "
            f"LR: {scheduler_backbone.get_last_lr()[0]:.6f}"
        )
        
        # Log to wandb
        if args.use_wandb:
            wandb.log({
                "epoch": epoch + 1,
                "train_total_loss": avg_total_loss,
                "train_cls_loss": avg_cls_loss,
                "train_domain_loss": avg_domain_loss,
                "train_source_acc": avg_source_acc,
                "train_domain_acc": avg_domain_acc,
                "learning_rate": scheduler_backbone.get_last_lr()[0]
            })
        
        # Save best model
        if avg_source_acc > best_acc:
            best_acc = avg_source_acc
            if not os.path.exists(args.checkpoint_dir):
                os.makedirs(args.checkpoint_dir)
            
            torch.save({
                'epoch': epoch + 1,
                'backbone_state_dict': backbone.state_dict(),
                'attention_state_dict': attention.state_dict(),
                'classifier_state_dict': classifier.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'best_acc': best_acc,
                'args': args
            }, os.path.join(args.checkpoint_dir, f'best_model_{args.source_domain}2{args.target_domain}.pth'))
            
            logger.info(f"New best accuracy: {best_acc:.2f}% | Saved checkpoint")
    
    # Final training stats
    avg_epoch_time = total_time / args.epochs
    logger.info("=== Training Completed ===")
    logger.info(f"Total training time: {total_time:.2f}s")
    logger.info(f"Average epoch time: {avg_epoch_time:.2f}s")
    logger.info(f"Best source accuracy: {best_acc:.2f}%")
    
    if args.use_wandb:
        wandb.finish()

def main():
    parser = argparse.ArgumentParser(description='RDAUP Training')
    
    # Dataset parameters
    parser.add_argument('--source_root', type=str, required=True, help='Source domain root directory')
    parser.add_argument('--target_root', type=str, required=True, help='Target domain root directory')
    parser.add_argument('--source_domain', type=str, required=True, help='Source domain name')
    parser.add_argument('--target_domain', type=str, required=True, help='Target domain name')
    
    # Model parameters
    parser.add_argument('--model_name', type=str, default='resnet50', help='Backbone model name')
    parser.add_argument('--num_classes', type=int, default=31, help='Number of classes')
    parser.add_argument('--freeze_until', type=int, default=1, help='Freeze backbone layers until index')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--lambda_adv', type=float, default=1.0, help='Weight for adversarial loss')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    # Logging and saving parameters
    parser.add_argument('--log_interval', type=int, default=10, help='Log interval')
    parser.add_argument('--checkpoint_dir', type=str, default='experiments/checkpoints', help='Checkpoint directory')
    parser.add_argument('--log_dir', type=str, default='experiments/logs', help='Log directory')
    parser.add_argument('--use_wandb', action='store_true', help='Use wandb for logging')
    
    args = parser.parse_args()
    
    # Set up logger
    logger = setup_logger(args.log_dir)
    
    # Start training
    train(args, logger)

if __name__ == '__main__':
    main()
