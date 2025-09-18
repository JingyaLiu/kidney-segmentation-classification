import logging
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import Counter


def setup_logging(output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = output_dir / 'training.log'
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    tensorboard_dir = output_dir / 'tensorboard'
    tensorboard_dir.mkdir(exist_ok=True)
    writer = SummaryWriter(tensorboard_dir)
    
    logger.info(f"Logging setup complete:")
    logger.info(f"  - Log file: {log_file}")
    logger.info(f"  - TensorBoard logs: {tensorboard_dir}")
    logger.info(f"  - Run TensorBoard with: tensorboard --logdir={tensorboard_dir}")
    
    return logger, writer


class MetricsTracker:
    def __init__(self, writer, logger, class_names=None):
        self.writer = writer
        self.logger = logger
        self.class_names = class_names or [f'Class_{i}' for i in range(5)]
        self.metrics_history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [],
            'learning_rate': [], 'epoch_time': []
        }
        self.best_val_acc = 0.0
        self.best_epoch = 0
        
    def log_epoch_metrics(self, epoch, train_loss, train_acc, val_loss, val_acc, 
                         learning_rate, epoch_time, model=None):
        self.metrics_history['train_loss'].append(train_loss)
        self.metrics_history['train_acc'].append(train_acc)
        self.metrics_history['val_loss'].append(val_loss)
        self.metrics_history['val_acc'].append(val_acc)
        self.metrics_history['learning_rate'].append(learning_rate)
        self.metrics_history['epoch_time'].append(epoch_time)
        
        self.writer.add_scalar('Loss/Train', train_loss, epoch)
        self.writer.add_scalar('Loss/Validation', val_loss, epoch)
        self.writer.add_scalar('Accuracy/Train', train_acc, epoch)
        self.writer.add_scalar('Accuracy/Validation', val_acc, epoch)
        self.writer.add_scalar('Learning_Rate', learning_rate, epoch)
        self.writer.add_scalar('Time/Epoch', epoch_time, epoch)
        
        self.logger.info(f"Epoch {epoch:3d}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        self.logger.info(f"          Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        self.logger.info(f"          LR: {learning_rate:.6f}, Time: {epoch_time:.1f}s")
        
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            self.best_epoch = epoch
            self.logger.info(f"          New best validation accuracy: {val_acc:.2f}%")
            
        if model and hasattr(model, 'get_feature_weights'):
            weights = model.get_feature_weights()
            self.writer.add_scalar('Model/Global_Weight', weights['global_weight'], epoch)
            self.writer.add_scalar('Model/Local_Weight', weights['local_weight'], epoch)
    
    def log_confusion_matrix(self, y_true, y_pred, epoch, phase='Validation'):
        cm = confusion_matrix(y_true, y_pred)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               xticklabels=self.class_names,
               yticklabels=self.class_names,
               title=f'{phase} Confusion Matrix (Epoch {epoch})',
               ylabel='True Label',
               xlabel='Predicted Label')
        
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'),
                       ha="center", va="center",
                       color="white" if cm[i, j] > thresh else "black")
        
        plt.tight_layout()
        self.writer.add_figure(f'Confusion_Matrix/{phase}', fig, epoch)
        plt.close(fig)
    
    def log_classification_report(self, y_true, y_pred, epoch, phase='Validation'):
        report = classification_report(y_true, y_pred, target_names=self.class_names, output_dict=True)
        
        for class_name in self.class_names:
            if class_name in report:
                self.writer.add_scalar(f'Per_Class/{phase}_Precision_{class_name}', 
                                     report[class_name]['precision'], epoch)
                self.writer.add_scalar(f'Per_Class/{phase}_Recall_{class_name}', 
                                     report[class_name]['recall'], epoch)
                self.writer.add_scalar(f'Per_Class/{phase}_F1_{class_name}', 
                                     report[class_name]['f1-score'], epoch)
        
        self.writer.add_scalar(f'Overall/{phase}_Macro_Avg_Precision', 
                             report['macro avg']['precision'], epoch)
        self.writer.add_scalar(f'Overall/{phase}_Macro_Avg_Recall', 
                             report['macro avg']['recall'], epoch)
        self.writer.add_scalar(f'Overall/{phase}_Macro_Avg_F1', 
                             report['macro avg']['f1-score'], epoch)
        self.writer.add_scalar(f'Overall/{phase}_Weighted_Avg_Precision', 
                             report['weighted avg']['precision'], epoch)
        self.writer.add_scalar(f'Overall/{phase}_Weighted_Avg_Recall', 
                             report['weighted avg']['recall'], epoch)
        self.writer.add_scalar(f'Overall/{phase}_Weighted_Avg_F1', 
                             report['weighted avg']['f1-score'], epoch)
    
    def log_training_summary(self, total_epochs, total_time):
        self.logger.info("=" * 60)
        self.logger.info("TRAINING SUMMARY")
        self.logger.info("=" * 60)
        self.logger.info(f"Total epochs: {total_epochs}")
        self.logger.info(f"Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
        self.logger.info(f"Best validation accuracy: {self.best_val_acc:.2f}% (epoch {self.best_epoch})")
        self.logger.info(f"Final training accuracy: {self.metrics_history['train_acc'][-1]:.2f}%")
        self.logger.info(f"Final validation accuracy: {self.metrics_history['val_acc'][-1]:.2f}%")
        self.logger.info("=" * 60)
        
        self.writer.add_scalar('Summary/Best_Val_Accuracy', self.best_val_acc, total_epochs)
        self.writer.add_scalar('Summary/Final_Train_Accuracy', self.metrics_history['train_acc'][-1], total_epochs)
        self.writer.add_scalar('Summary/Final_Val_Accuracy', self.metrics_history['val_acc'][-1], total_epochs)
        self.writer.add_scalar('Summary/Total_Time_Minutes', total_time/60, total_epochs)
    
    def close(self):
        self.writer.close()


def create_data_loaders(train_dataset, val_dataset, args, labels=None):
    if args.imbalance_method in ['sampler', 'both'] and labels is not None:
        class_sample_counts = np.array([Counter(labels)[i] for i in range(args.num_classes)])
        weights = 1. / class_sample_counts
        samples_weight = np.array([weights[labels[i]] for i in range(len(labels))])
        samples_weight = torch.from_numpy(samples_weight).double()
        sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=args.batch_size, 
            sampler=sampler, 
            num_workers=args.num_workers, 
            pin_memory=True
        )
    else:
        train_loader = DataLoader(
            train_dataset, 
            batch_size=args.batch_size, 
            shuffle=True, 
            num_workers=args.num_workers, 
            pin_memory=True
        )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers, 
        pin_memory=True
    )
    
    return train_loader, val_loader


def create_optimizer_and_scheduler(model, args):
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    if args.scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=args.num_epochs
        )
    elif args.scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, 
            step_size=args.num_epochs // 3, 
            gamma=0.1
        )
    else:
        scheduler = None
    
    return optimizer, scheduler


def train_epoch(model, train_loader, criterion, optimizer, device, args):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []
    
    progress_bar = tqdm(train_loader, desc='Training')
    
    for batch_idx, batch in enumerate(progress_bar):
        if isinstance(batch, dict):
            if 'kidney_tumor_roi' in batch:
                kidney_tumor_roi = batch['kidney_tumor_roi'].to(device)
                tumor_roi = batch['tumor_roi'].to(device)
                labels = batch['label'].to(device)
            else:
                images = batch['images'].to(device)
                labels = batch['labels'].to(device)
                kidney_tumor_roi = images
                tumor_roi = None
        else:
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)
            kidney_tumor_roi = images
            tumor_roi = None
        
        optimizer.zero_grad()
        if tumor_roi is not None:
            outputs = model(kidney_tumor_roi, tumor_roi)
        else:
            outputs = model(kidney_tumor_roi)
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
    
    avg_loss = total_loss / len(train_loader)
    accuracy = 100 * correct / total
    
    return avg_loss, accuracy, all_predictions, all_labels


def validate_epoch(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Validation'):
            if isinstance(batch, dict):
                if 'kidney_tumor_roi' in batch:
                    kidney_tumor_roi = batch['kidney_tumor_roi'].to(device)
                    tumor_roi = batch['tumor_roi'].to(device)
                    labels = batch['label'].to(device)
                else:
                    images = batch['images'].to(device)
                    labels = batch['labels'].to(device)
                    kidney_tumor_roi = images
                    tumor_roi = None
            else:
                images, labels = batch
                images = images.to(device)
                labels = labels.to(device)
                kidney_tumor_roi = images
                tumor_roi = None
            
            if tumor_roi is not None:
                outputs = model(kidney_tumor_roi, tumor_roi)
            else:
                outputs = model(kidney_tumor_roi)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(val_loader)
    accuracy = 100 * correct / total
    
    return avg_loss, accuracy, all_predictions, all_labels


def plot_training_curves(train_losses, val_losses, train_accs, val_accs, output_dir, fold):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    ax1.plot(train_losses, label='Training Loss', color='blue')
    ax1.plot(val_losses, label='Validation Loss', color='red')
    ax1.set_title(f'Training and Validation Loss - Fold {fold}')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(train_accs, label='Training Accuracy', color='blue')
    ax2.plot(val_accs, label='Validation Accuracy', color='red')
    ax2.set_title(f'Training and Validation Accuracy - Fold {fold}')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_dir / f'training_curves_fold_{fold}.png', dpi=300, bbox_inches='tight')
    plt.close()


def save_model_checkpoint(model, optimizer, epoch, val_accuracy, output_dir, fold, is_best=False):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_accuracy': val_accuracy
    }
    
    torch.save(checkpoint, output_dir / f'latest_model_fold_{fold}.pth')
    
    if is_best:
        torch.save(checkpoint, output_dir / f'best_model_fold_{fold}.pth')


def load_model_checkpoint(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    return checkpoint
