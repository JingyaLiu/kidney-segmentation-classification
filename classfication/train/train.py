#!/usr/bin/env python3

import os
import sys
import json
import argparse
import numpy as np
import torch
import time
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.dataset import MultiSliceDataset
from utils.augmentation import get_multi_slice_transforms
from utils.data_split import create_train_test_split, load_train_test_split
from utils.training import (
    setup_logging, create_data_loaders, create_optimizer_and_scheduler,
    train_epoch, validate_epoch, plot_training_curves, save_model_checkpoint,
    MetricsTracker
)
from utils.losses import calculate_class_weights, get_loss_function
from model.models import MedicalDualPathNetwork


def parse_args():
    parser = argparse.ArgumentParser(description='Train dual-path classification')
    
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--output_dir', type=str, default='./results/model')
    parser.add_argument('--model_weights_dir', type=str, default='./model_weights/model')
    parser.add_argument('--num_slices', type=int, default=3)
    parser.add_argument('--test_size', type=float, default=0.2)
    parser.add_argument('--splits_file', type=str, default='data_splits.json')
    
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--dropout_rate', type=float, default=0.3)
    
    parser.add_argument('--pretrained_type', type=str, default='radimagenet', 
                       choices=['imagenet', 'radimagenet', 'none'])
    parser.add_argument('--pretrained_global_path', type=str, default=None)
    parser.add_argument('--pretrained_local_path', type=str, default=None)
    parser.add_argument('--pool_type', type=str, default='attn', choices=['mean', 'attn'])
    parser.add_argument('--freeze_until', type=str, default='layer4',
                       choices=['layer2', 'layer3', 'layer4', 'none'])
    
    parser.add_argument('--use_masks', action='store_true', default=True)
    parser.add_argument('--input_channels', type=int, default=4)
    
    parser.add_argument('--num_folds', type=int, default=5)
    parser.add_argument('--scheduler', type=str, default='cosine', choices=['cosine', 'step', 'none'])
    parser.add_argument('--grad_clip', type=float, default=1.0)
    
    parser.add_argument('--imbalance_method', type=str, default='both', 
                       choices=['sampler', 'weights', 'both', 'none'])
    parser.add_argument('--loss_type', type=str, default='cross_entropy', 
                       choices=['cross_entropy', 'focal', 'label_smoothing'])
    parser.add_argument('--focal_alpha', type=float, default=0.25)
    parser.add_argument('--focal_gamma', type=float, default=2.0)
    
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--seed', type=int, default=42)
    
    return parser.parse_args()


def setup_device(args):
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")
    return device


def setup_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def prepare_data(kidney_tumor_roi, tumor_roi, use_masks=True):
    B, S, _, H, W = kidney_tumor_roi.shape
    
    if use_masks:
        global_x = kidney_tumor_roi.repeat(1, 1, 3, 1, 1)
        global_x = torch.cat([global_x, tumor_roi], dim=2)
        local_x = tumor_roi.repeat(1, 1, 3, 1, 1)
        local_x = torch.cat([local_x, tumor_roi], dim=2)
    else:
        global_x = kidney_tumor_roi.repeat(1, 1, 3, 1, 1)
        local_x = tumor_roi.repeat(1, 1, 3, 1, 1)
    
    slice_mask = torch.ones(B, S, dtype=torch.bool)
    return global_x, local_x, slice_mask


def train_epoch(model, train_loader, criterion, optimizer, device, args):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []
    
    progress_bar = tqdm(train_loader, desc='Training')
    
    for batch_idx, batch in enumerate(progress_bar):
        kidney_tumor_roi = batch['kidney_tumor_roi'].to(device)
        tumor_roi = batch['tumor_roi'].to(device)
        labels = batch['label'].to(device)
        
        global_x, local_x, slice_mask = prepare_data(
            kidney_tumor_roi, tumor_roi, use_masks=args.use_masks
        )
        slice_mask = slice_mask.to(device)
        
        optimizer.zero_grad()
        outputs = model(global_x, local_x, slice_mask)
        loss = criterion(outputs, labels)
        loss.backward()
        
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        all_predictions.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        progress_bar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Acc': f'{100 * correct / total:.2f}%'
        })
    
    return total_loss / len(train_loader), 100 * correct / total, all_predictions, all_labels


def validate_epoch(model, val_loader, criterion, device, args):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Validation'):
            kidney_tumor_roi = batch['kidney_tumor_roi'].to(device)
            tumor_roi = batch['tumor_roi'].to(device)
            labels = batch['label'].to(device)
            
            global_x, local_x, slice_mask = prepare_data(
                kidney_tumor_roi, tumor_roi, use_masks=args.use_masks
            )
            slice_mask = slice_mask.to(device)
            
            outputs = model(global_x, local_x, slice_mask)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return total_loss / len(val_loader), 100 * correct / total, all_predictions, all_labels


def main():
    args = parse_args()
    
    setup_seed(args.seed)
    device = setup_device(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model_weights_dir = Path(args.model_weights_dir)
    model_weights_dir.mkdir(parents=True, exist_ok=True)
    
    logger, writer = setup_logging(output_dir)
    logger.info(f"Starting training with args: {vars(args)}")
    
    class_names = ['Clear Cell RCC', 'Papillary RCC', 'Chromophobe RCC', 'Oncocytoma', 'Other']
    metrics_tracker = MetricsTracker(writer, logger, class_names)
    
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    logger.info("Loading dataset...")
    dataset = MultiSliceDataset(
        data_dir=args.data_dir,
        transform=get_multi_slice_transforms(is_training=True, num_slices=args.num_slices),
        split='train',
        num_slices=args.num_slices
    )
    
    labels = [dataset[i]['label'] for i in range(len(dataset))]
    
    train_indices, test_indices = load_train_test_split(args.splits_file)
    if train_indices is None:
        logger.info(f"Creating new train/test split and saving to {args.splits_file}...")
        train_indices, test_indices = create_train_test_split(labels, args.test_size, args.seed, args.splits_file)
    else:
        logger.info(f"Loaded existing train/test split from {args.splits_file}")
    
    dataset.cases_data = [dataset.cases_data[i] for i in train_indices]
    labels = [labels[i] for i in train_indices]
    
    args.num_classes = len(set(labels))
    
    logger.info(f"Dataset loaded: {len(dataset)} cases, {args.num_classes} classes")
    logger.info(f"Class distribution: {dict(zip(*np.unique(labels, return_counts=True)))}")
    
    skf = StratifiedKFold(n_splits=args.num_folds, shuffle=True, random_state=args.seed)
    
    all_results = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(range(len(dataset)), labels), 1):
        logger.info(f"\n{'='*50}")
        logger.info(f"Training Fold {fold}/{args.num_folds}")
        logger.info(f"{'='*50}")
        
        train_dataset = MultiSliceDataset(
            data_dir=args.data_dir,
            transform=get_multi_slice_transforms(is_training=True, num_slices=args.num_slices),
            split='train',
            num_slices=args.num_slices
        )
        val_dataset = MultiSliceDataset(
            data_dir=args.data_dir,
            transform=get_multi_slice_transforms(is_training=False, num_slices=args.num_slices),
            split='val',
            num_slices=args.num_slices
        )
        
        train_dataset.cases_data = [dataset.cases_data[i] for i in train_idx]
        val_dataset.cases_data = [dataset.cases_data[i] for i in val_idx]
        
        train_loader, val_loader = create_data_loaders(
            train_dataset, val_dataset, args, 
            labels=[labels[i] for i in train_idx]
        )
        
        model = MedicalDualPathNetwork(
            num_classes=args.num_classes,
            num_slices=args.num_slices,
            gin_channels=args.input_channels,
            lin_channels=args.input_channels,
            pretrained_type=args.pretrained_type,
            pretrained_global_path=args.pretrained_global_path,
            pretrained_local_path=args.pretrained_local_path,
            pool=args.pool_type,
            dropout=args.dropout_rate,
            freeze_until=args.freeze_until,
        ).to(device)
        
        logger.info(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
        logger.info(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        
        if args.imbalance_method in ['weights', 'both']:
            class_weights = calculate_class_weights([labels[i] for i in train_idx])
            criterion = get_loss_function(
                loss_type=args.loss_type,
                class_weights=class_weights.to(device),
                focal_alpha=args.focal_alpha,
                focal_gamma=args.focal_gamma
            )
        else:
            criterion = get_loss_function(
                loss_type=args.loss_type,
                focal_alpha=args.focal_alpha,
                focal_gamma=args.focal_gamma
            )
        
        optimizer, scheduler = create_optimizer_and_scheduler(model, args)
        
        best_val_acc = 0.0
        train_losses, val_losses = [], []
        train_accs, val_accs = [], []
        
        for epoch in range(args.num_epochs):
            epoch_start_time = time.time()
            
            train_loss, train_acc, train_preds, train_labels = train_epoch(
                model, train_loader, criterion, optimizer, device, args
            )
            
            val_loss, val_acc, val_preds, val_labels = validate_epoch(
                model, val_loader, criterion, device, args
            )
            
            if scheduler:
                scheduler.step()
            
            epoch_time = time.time() - epoch_start_time
            current_lr = optimizer.param_groups[0]['lr']
            
            metrics_tracker.log_epoch_metrics(
                epoch + 1, train_loss, train_acc, val_loss, val_acc,
                current_lr, epoch_time, model
            )
            
            if (epoch + 1) % 10 == 0:
                metrics_tracker.log_confusion_matrix(val_labels, val_preds, epoch + 1, 'Validation')
                metrics_tracker.log_classification_report(val_labels, val_preds, epoch + 1, 'Validation')
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accs.append(train_acc)
            val_accs.append(val_acc)
            
            is_best = val_acc > best_val_acc
            if is_best:
                best_val_acc = val_acc
                logger.info(f"New best validation accuracy: {best_val_acc:.2f}%")
            
            save_model_checkpoint(
                model, optimizer, epoch, val_acc, model_weights_dir, fold, is_best
            )
        
        plot_training_curves(
            train_losses, val_losses, train_accs, val_accs, output_dir, fold
        )
        
        total_training_time = sum(metrics_tracker.metrics_history['epoch_time'])
        metrics_tracker.log_training_summary(args.num_epochs, total_training_time)
        
        metrics_tracker.log_confusion_matrix(val_labels, val_preds, args.num_epochs, 'Final_Validation')
        metrics_tracker.log_classification_report(val_labels, val_preds, args.num_epochs, 'Final_Validation')
        
        val_preds = np.array(val_preds)
        val_labels = np.array(val_labels)
        
        try:
            auc = roc_auc_score(val_labels, val_preds, multi_class='ovr', average='macro')
            logger.info(f"Validation AUC: {auc:.4f}")
        except:
            logger.info("Validation AUC: N/A (insufficient data)")
        
        all_results.append({
            'fold': fold,
            'best_val_acc': best_val_acc,
            'final_train_acc': train_accs[-1],
            'final_val_acc': val_accs[-1],
            'total_training_time': total_training_time
        })
        
        metrics_tracker.close()
    
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    avg_val_acc = np.mean([r['best_val_acc'] for r in all_results])
    std_val_acc = np.std([r['best_val_acc'] for r in all_results])
    
    logger.info(f"Training completed - Validation Accuracy: {avg_val_acc:.2f}% ± {std_val_acc:.2f}%")
    logger.info(f"Results saved to: {output_dir}")
    
    return all_results


if __name__ == '__main__':
    main()
