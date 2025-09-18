import os
import sys
import json
import argparse
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import seaborn as sns

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.models import MedicalDualPathNetwork


def load_model(model_path, device, num_classes=5, num_slices=4, gin_channels=4, lin_channels=4):
    """Load a trained model from checkpoint."""
    try:
        checkpoint = torch.load(model_path, map_location=device)
        
        model = MedicalDualPathNetwork(
            num_classes=num_classes,
            num_slices=num_slices,
            gin_channels=gin_channels,
            lin_channels=lin_channels,
            pretrained_type='imagenet',
            pool='attn'
        ).to(device)
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.eval()
        return model
        
    except Exception as e:
        print(f"Error loading model from {model_path}: {e}")
        return None


def prepare_multi_channel_data(kidney_tumor_roi, tumor_roi, use_masks=True):
    """Prepare multi-channel input data by concatenating CT slices and masks."""
    B, S, _, H, W = kidney_tumor_roi.shape
    
    if use_masks:
        global_x = kidney_tumor_roi.repeat(1, 1, 3, 1, 1)
        global_x = torch.cat([global_x, tumor_roi], dim=2)
        
        local_x = tumor_roi.repeat(1, 1, 3, 1, 1)
        local_x = torch.cat([local_x, tumor_roi], dim=2)
    else:
        global_x = kidney_tumor_roi.repeat(1, 1, 3, 1, 1)
        local_x = tumor_roi.repeat(1, 1, 3, 1, 1)
    
    slice_mask = torch.ones(B, S, device=kidney_tumor_roi.device, dtype=torch.bool)
    return global_x, local_x, slice_mask


def map_label_to_model_classes(label, class_name):
    """Map test case labels to model class indices."""
    class_mapping = {
        'Clear Cell': 0,
        'Papillary': 1,
        'Chromophobe': 2
    }
    
    return class_mapping.get(class_name, label)


def load_test_case_data(data_dir, case_id, device):
    """Load data for a specific test case."""
    case_dir = Path(data_dir) / case_id
    
    kidney_tumor_roi = np.load(case_dir / 'kidney_tumor_roi.npy')
    tumor_roi = np.load(case_dir / 'tumor_roi.npy')
    
    with open(case_dir / 'metadata.json', 'r') as f:
        metadata = json.load(f)
    
    kidney_tumor_roi = torch.from_numpy(kidney_tumor_roi).float().unsqueeze(0).unsqueeze(2)
    tumor_roi = torch.from_numpy(tumor_roi).float().unsqueeze(0).unsqueeze(2)
    
    kidney_tumor_roi = kidney_tumor_roi.to(device)
    tumor_roi = tumor_roi.to(device)
    
    return kidney_tumor_roi, tumor_roi, metadata


def ensemble_predict_single_case(models, kidney_tumor_roi, tumor_roi, device, use_masks=True):
    """Perform ensemble prediction on a single case."""
    global_x, local_x, slice_mask = prepare_multi_channel_data(
        kidney_tumor_roi, tumor_roi, use_masks=use_masks
    )
    
    fold_probabilities = []
    
    with torch.no_grad():
        for model in models:
            if model is not None:
                outputs = model(global_x, local_x, slice_mask)
                probabilities = torch.softmax(outputs, dim=1)
                fold_probabilities.append(probabilities.cpu().numpy())
    
    if fold_probabilities:
        avg_probabilities = np.mean(fold_probabilities, axis=0)
        final_prediction = np.argmax(avg_probabilities, axis=1)[0]
        return final_prediction, avg_probabilities[0]
    else:
        return None, None


def apply_threshold(probabilities, threshold=0.5, target_class=0):
    """Apply threshold to probabilities for binary classification."""
    target_prob = probabilities[target_class]
    binary_prediction = int(target_prob >= threshold)
    
    return binary_prediction, target_prob


def calculate_sensitivity_specificity(y_true, y_pred, class_names, unique_classes):
    """Calculate sensitivity and specificity for each class."""
    cm = confusion_matrix(y_true, y_pred, labels=unique_classes)
    sensitivity_specificity = {}
    
    for i, class_idx in enumerate(unique_classes):
        if class_idx < len(class_names):
            class_name = class_names[class_idx]
            
            # True Positives: correctly predicted as class i
            tp = cm[i, i]
            # False Negatives: actually class i but predicted as other classes
            fn = cm[i, :].sum() - tp
            # False Positives: not class i but predicted as class i
            fp = cm[:, i].sum() - tp
            # True Negatives: not class i and correctly predicted as not class i
            tn = cm.sum() - (tp + fn + fp)
            
            # Sensitivity (Recall) = TP / (TP + FN)
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            
            # Specificity = TN / (TN + FP)
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            
            sensitivity_specificity[f'{class_name}_sensitivity'] = sensitivity
            sensitivity_specificity[f'{class_name}_specificity'] = specificity
    
    return sensitivity_specificity


def calculate_metrics(y_true, y_pred, y_prob, class_names):
    """Calculate comprehensive evaluation metrics."""
    metrics = {}
    
    unique_classes = np.unique(np.concatenate([y_true, y_pred]))
    present_class_names = [class_names[i] for i in unique_classes if i < len(class_names)]
    
    print(f"Unique classes in data: {unique_classes}")
    print(f"Present class names: {present_class_names}")
    
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    
    if len(present_class_names) > 0:
        report = classification_report(y_true, y_pred, labels=unique_classes, target_names=present_class_names, output_dict=True)
        metrics['classification_report'] = report
        
        for i, class_name in enumerate(present_class_names):
            if class_name in report:
                metrics[f'{class_name}_precision'] = report[class_name]['precision']
                metrics[f'{class_name}_recall'] = report[class_name]['recall']
                metrics[f'{class_name}_f1'] = report[class_name]['f1-score']
    else:
        metrics['classification_report'] = {}
    
    cm = confusion_matrix(y_true, y_pred, labels=unique_classes)
    metrics['confusion_matrix'] = cm.tolist()
    metrics['unique_classes'] = unique_classes.tolist()
    metrics['present_class_names'] = present_class_names
    
    # Calculate sensitivity and specificity
    sens_spec = calculate_sensitivity_specificity(y_true, y_pred, class_names, unique_classes)
    metrics.update(sens_spec)
    
    try:
        metrics['macro_auc'] = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')
        metrics['weighted_auc'] = roc_auc_score(y_true, y_prob, multi_class='ovr', average='weighted')
    except:
        metrics['macro_auc'] = None
        metrics['weighted_auc'] = None
    
    return metrics


def plot_confusion_matrix(cm, class_names, output_path, title="Confusion Matrix"):
    """Plot and save confusion matrix."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_class_probabilities(probabilities, labels, class_names, output_path):
    """Plot class probability distributions."""
    n_classes = len(class_names)
    n_cols = min(3, n_classes)
    n_rows = (n_classes + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    if n_classes == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes if isinstance(axes, list) else [axes]
    else:
        axes = axes.flatten()
    
    for i, class_name in enumerate(class_names):
        class_probs = probabilities[:, i]
        axes[i].hist(class_probs, bins=20, alpha=0.7, edgecolor='black')
        axes[i].set_title(f'{class_name} Probability Distribution')
        axes[i].set_xlabel('Probability')
        axes[i].set_ylabel('Frequency')
        axes[i].grid(True, alpha=0.3)
    
    for i in range(n_classes, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def print_inference_results(metrics, class_names, clear_cell_threshold, clear_cell_accuracy):
    """Print formatted inference results."""
    print("\n" + "="*60)
    print("TEST CASES INFERENCE RESULTS")
    print("="*60)
    print(f"Overall Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro AUC: {metrics['macro_auc']:.4f}" if metrics['macro_auc'] else "Macro AUC: N/A")
    print(f"Weighted AUC: {metrics['weighted_auc']:.4f}" if metrics['weighted_auc'] else "Weighted AUC: N/A")
    
    print(f"\nClear Cell RCC Detection (threshold={clear_cell_threshold}):")
    print(f"  Binary Accuracy: {clear_cell_accuracy:.4f}")
    
    print(f"\nPer-Class Performance:")
    present_class_names = metrics['present_class_names']
    for class_name in present_class_names:
        if f'{class_name}_precision' in metrics:
            print(f"  {class_name}:")
            print(f"    Precision: {metrics[f'{class_name}_precision']:.4f}")
            print(f"    Recall (Sensitivity): {metrics[f'{class_name}_recall']:.4f}")
            print(f"    Specificity: {metrics[f'{class_name}_specificity']:.4f}")
            print(f"    F1-Score: {metrics[f'{class_name}_f1']:.4f}")


def main():
    parser = argparse.ArgumentParser(description='Inference on processed test cases with orientation corrections')
    
    parser.add_argument('--data_dir', type=str, 
                       default='./processed_test_cases_with_orientation_corrections',
                       help='Path to processed test cases directory')
    parser.add_argument('--model_weights_dir', type=str, default='./model_weights',
                       help='Directory containing model weights')
    parser.add_argument('--output_dir', type=str, default='./results/test_cases_inference',
                       help='Output directory for inference results')
    
    parser.add_argument('--folds', type=int, nargs='+', default=[1, 2, 3, 4, 5],
                       help='Fold numbers to use for ensemble')
    parser.add_argument('--num_classes', type=int, default=5,
                       help='Number of classes')
    parser.add_argument('--num_slices', type=int, default=4,
                       help='Number of slices per case')
    parser.add_argument('--gin_channels', type=int, default=4,
                       help='Number of global input channels per slice (3 CT + 1 mask)')
    parser.add_argument('--lin_channels', type=int, default=4,
                       help='Number of local input channels per slice (3 CT + 1 mask)')
    parser.add_argument('--use_masks', action='store_true', default=True,
                       help='Use tumor masks as additional channel')
    
    parser.add_argument('--clear_cell_threshold', type=float, default=0.5,
                       help='Threshold for clear cell RCC detection')
    
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cpu, cuda)')
    
    args = parser.parse_args()
    
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    class_names = ['Clear Cell RCC', 'Papillary RCC', 'Chromophobe RCC', 'Oncocytoma', 'Other']
    
    summary_path = Path(args.data_dir) / 'summary.json'
    with open(summary_path, 'r') as f:
        summary = json.load(f)
    
    print(f"Model classes: {class_names}")
    print(f"Test data class distribution: {summary['class_distribution']}")
    
    print("Loading ensemble models...")
    models = []
    for fold in args.folds:
        model_path = Path(args.model_weights_dir) / f'best_model_fold_{fold}.pth'
        if model_path.exists():
            model = load_model(model_path, device, args.num_classes, args.num_slices, args.gin_channels, args.lin_channels)
            if model is not None:
                models.append(model)
                print(f"  Loaded fold {fold} model")
            else:
                print(f"  Failed to load fold {fold} model")
        else:
            print(f"  Model not found: {model_path}")
    
    if not models:
        print("No models loaded successfully!")
        return
    
    print(f"Loaded {len(models)} models for ensemble")
    
    case_ids = summary['processed_cases']
    print(f"Found {len(case_ids)} test cases")
    
    print("Performing ensemble inference...")
    all_predictions = []
    all_probabilities = []
    all_labels = []
    all_case_ids = []
    all_metadata = []
    
    for case_id in tqdm(case_ids, desc="Processing cases"):
        try:
            kidney_tumor_roi, tumor_roi, metadata = load_test_case_data(args.data_dir, case_id, device)
            
            prediction, probabilities = ensemble_predict_single_case(
                models, kidney_tumor_roi, tumor_roi, device, args.use_masks
            )
            
            if prediction is not None:
                all_predictions.append(prediction)
                all_probabilities.append(probabilities)
                mapped_label = map_label_to_model_classes(metadata['label'], metadata['class_name'])
                all_labels.append(mapped_label)
                all_case_ids.append(case_id)
                all_metadata.append(metadata)
            else:
                print(f"Failed to predict case: {case_id}")
                
        except Exception as e:
            print(f"Error processing case {case_id}: {e}")
            continue
    
    all_predictions = np.array(all_predictions)
    all_probabilities = np.array(all_probabilities)
    all_labels = np.array(all_labels)
    
    print(f"Successfully processed {len(all_predictions)} cases")
    
    print(f"Applying threshold {args.clear_cell_threshold} for Clear Cell RCC detection...")
    binary_predictions = []
    clear_cell_probs = []
    
    for prob in all_probabilities:
        binary_pred, clear_cell_prob = apply_threshold(prob, args.clear_cell_threshold, target_class=0)
        binary_predictions.append(binary_pred)
        clear_cell_probs.append(clear_cell_prob)
    
    binary_predictions = np.array(binary_predictions)
    clear_cell_probs = np.array(clear_cell_probs)
    
    print("Calculating metrics...")
    metrics = calculate_metrics(all_labels, all_predictions, all_probabilities, class_names)
    
    clear_cell_accuracy = accuracy_score((all_labels == 0).astype(int), binary_predictions)
    print_inference_results(metrics, class_names, args.clear_cell_threshold, clear_cell_accuracy)
    
    print("\nSaving results...")
    
    with open(output_dir / 'test_cases_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    np.save(output_dir / 'test_cases_predictions.npy', all_predictions)
    np.save(output_dir / 'test_cases_probabilities.npy', all_probabilities)
    np.save(output_dir / 'test_cases_labels.npy', all_labels)
    np.save(output_dir / 'test_cases_binary_predictions.npy', binary_predictions)
    np.save(output_dir / 'test_cases_clear_cell_probabilities.npy', clear_cell_probs)
    
    results = []
    for i in range(len(all_labels)):
        result = {
            'case_id': all_case_ids[i],
            'true_label': int(all_labels[i]),
            'true_class': class_names[int(all_labels[i])],
            'predicted_label': int(all_predictions[i]),
            'predicted_class': class_names[int(all_predictions[i])],
            'clear_cell_probability': float(clear_cell_probs[i]),
            'binary_prediction': int(binary_predictions[i]),
            'class_probabilities': {
                class_names[j]: float(all_probabilities[i, j]) for j in range(len(class_names))
            },
            'metadata': all_metadata[i]
        }
        results.append(result)
    
    with open(output_dir / 'test_cases_detailed_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("Creating visualizations...")
    
    present_class_names = metrics['present_class_names']
    
    cm = np.array(metrics['confusion_matrix'])
    plot_confusion_matrix(cm, present_class_names, output_dir / 'test_cases_confusion_matrix.png')
    
    plot_class_probabilities(all_probabilities, all_labels, present_class_names, output_dir / 'test_cases_class_probabilities.png')
    
    plt.figure(figsize=(10, 6))
    plt.hist(clear_cell_probs, bins=20, alpha=0.7, edgecolor='black')
    plt.axvline(args.clear_cell_threshold, color='red', linestyle='--', 
                label=f'Threshold = {args.clear_cell_threshold}')
    plt.xlabel('Clear Cell RCC Probability')
    plt.ylabel('Frequency')
    plt.title('Clear Cell RCC Probability Distribution (Test Cases)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'test_cases_clear_cell_probability_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nTest cases inference completed successfully!")
    print(f"Results saved to: {output_dir}")
    print(f"Visualizations: test_cases_confusion_matrix.png, test_cases_class_probabilities.png, test_cases_clear_cell_probability_distribution.png")
    print("="*60)


if __name__ == '__main__':
    main()