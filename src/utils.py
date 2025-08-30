"""
Utility Functions for Image Captioning Project
This file contains helper functions for:
1. Training utilities (save/load models, logging)
2. Evaluation metrics (BLEU score)
3. Visualization (plotting, showing results)
4. Model inference (generating captions for new images)
"""

import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from collections import Counter
import json
import pickle
from datetime import datetime
import math

class AverageMeter:
    """
    Computes and stores the average and current value
    Useful for tracking loss during training
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(model, optimizer, epoch, loss, accuracy, checkpoint_dir="checkpoints/"):
    """
    Save model checkpoint
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'accuracy': accuracy,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Saved epoch number
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
    torch.save(checkpoint, checkpoint_path)
    
    # Also saved as latest checkpoint
    latest_path = os.path.join(checkpoint_dir, 'checkpoint_latest.pth')
    torch.save(checkpoint, latest_path)
    print(f"Checkpoint saved: {checkpoint_path}")
    return checkpoint_path

def load_checkpoint(checkpoint_path, model, optimizer=None):
    """
    Load model checkpoint
    Args:
        checkpoint_path (str): Path to checkpoint file
        model: Model to load weights into
        optimizer: Optimizer to load state into (optional)
    Returns:
        dict: Checkpoint information
    """
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        return None
    
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Loading model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Loading optimizer state if provided
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print(f"Checkpoint loaded successfully!")
    print(f"Epoch: {checkpoint.get('epoch', 'Unknown')}")
    print(f"Loss: {checkpoint.get('loss', 'Unknown'):.4f}")
    print(f"Accuracy: {checkpoint.get('accuracy', 'Unknown'):.4f}")
    return checkpoint


def calculate_bleu_score(predicted_captions, target_captions, max_n=4):
    """
    Calculate BLEU score for caption evaluation
    
    Args:
        predicted_captions (list): List of predicted captions (strings)
        target_captions (list): List of target captions (strings)
        max_n (int): Maximum n-gram order
        
    Returns:
        float: BLEU score (0-1)
    """
    def get_ngrams(tokens, n):
        """Get n-grams from token list"""
        ngrams = []
        for i in range(len(tokens) - n + 1):
            ngrams.append(tuple(tokens[i:i+n]))
        return ngrams
    
    if len(predicted_captions) != len(target_captions):
        print("Warning: Predicted and target caption lists have different lengths")
        min_len = min(len(predicted_captions), len(target_captions))
        predicted_captions = predicted_captions[:min_len]
        target_captions = target_captions[:min_len]
    
    bleu_scores = []
    
    for pred, target in zip(predicted_captions, target_captions):
        pred_tokens = pred.lower().split()
        target_tokens = target.lower().split()
        
        # Calculate precision for each n-gram order
        precisions = []
        
        for n in range(1, max_n + 1):
            pred_ngrams = get_ngrams(pred_tokens, n)
            target_ngrams = get_ngrams(target_tokens, n)
            
            if len(pred_ngrams) == 0:
                precisions.append(0.0)
                continue
            
            # Count matches
            pred_counter = Counter(pred_ngrams)
            target_counter = Counter(target_ngrams)
            
            matches = 0
            for ngram in pred_counter:
                matches += min(pred_counter[ngram], target_counter[ngram])
            
            precision = matches / len(pred_ngrams)
            precisions.append(precision)
        
        # Brevity penalty
        bp = min(1.0, len(pred_tokens) / max(len(target_tokens), 1))
        
        # Geometric mean of precisions
        if all(p > 0 for p in precisions):
            bleu = bp * math.exp(sum(math.log(p) for p in precisions) / len(precisions))
        else:
            bleu = 0.0
        
        bleu_scores.append(bleu)
    
    return sum(bleu_scores) / len(bleu_scores)


def plot_training_history(train_losses, val_losses, train_accuracies, val_accuracies, save_path=None):
    """
    Plot training history
    
    Args:
        train_losses (list): Training losses per epoch
        val_losses (list): Validation losses per epoch  
        train_accuracies (list): Training accuracies per epoch
        val_accuracies (list): Validation accuracies per epoch
        save_path (str): Path to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot losses
    epochs = range(1, len(train_losses) + 1)
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss')
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss')
    ax1.set_title('Model Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot accuracies
    ax2.plot(epochs, train_accuracies, 'b-', label='Training Accuracy')
    ax2.plot(epochs, val_accuracies, 'r-', label='Validation Accuracy')
    ax2.set_title('Model Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved: {save_path}")
    
    plt.show()


def generate_caption(model, image_path, vocabulary, transform, device, max_length=50):
    """
    Generate caption for a single image
    
    Args:
        model: Trained captioning model
        image_path (str): Path to image file
        vocabulary: Vocabulary object
        transform: Image preprocessing transform
        device: Device (CPU/GPU)
        max_length (int): Maximum caption length
        
    Returns:
        str: Generated caption
    """
    model.eval()
    
    # Load and preprocess image
    try:
        image = Image.open(image_path).convert('RGB')
        image = transform(image).unsqueeze(0).to(device)
    except Exception as e:
        print(f"Error loading image: {e}")
        return "Error: Could not load image"
    
    with torch.no_grad():
        # Get image features
        features = model.encoder(image)
        
        # Initialize caption with START token
        caption = [vocabulary.word_to_idx['<START>']]
        
        # Generate caption word by word
        hidden = model.decoder.init_hidden(1, device)
        
        for _ in range(max_length):
            # Convert to tensor
            input_word = torch.LongTensor([caption[-1]]).to(device)
            
            # Get word embedding
            embedded = model.decoder.embedding(input_word)
            
            # LSTM forward pass
            lstm_input = torch.cat([features, embedded], dim=1).unsqueeze(1)
            output, hidden = model.decoder.lstm(lstm_input, hidden)
            
            # Get word predictions
            output = model.decoder.fc(output.squeeze(1))
            predicted = output.argmax(1).item()
            caption.append(predicted)
            
            # Stop if END token
            if predicted == vocabulary.word_to_idx['<END>']:
                break
    
    # Convert to words
    caption_words = []
    for idx in caption[1:]:  # Skip START token
        word = vocabulary.idx_to_word[idx]
        if word == '<END>':
            break
        if word not in ['<PAD>', '<START>']:
            caption_words.append(word)
    
    return ' '.join(caption_words)


def visualize_predictions(model, data_loader, vocabulary, device, num_images=5, save_path=None):
    """
    Visualize model predictions on sample images
    
    Args:
        model: Trained model
        data_loader: Data loader
        vocabulary: Vocabulary object
        device: Device (CPU/GPU)
        num_images (int): Number of images to show
        save_path (str): Path to save the visualization
    """
    model.eval()
    
    #batch of data
    images, target_captions, lengths = next(iter(data_loader))
    
    # Selecting subset
    images = images[:num_images].to(device)
    target_captions = target_captions[:num_images]
    
    # To Generate predictions
    predicted_captions = []
    
    with torch.no_grad():
        for i in range(num_images):
            # Get single image
            single_image = images[i:i+1]
            
            # Get features
            features = model.encoder(single_image)
            
            # Generate caption
            caption = [vocabulary.word_to_idx['<START>']]
            hidden = model.decoder.init_hidden(1, device)
            
            for _ in range(50): 
                input_word = torch.LongTensor([caption[-1]]).to(device)
                embedded = model.decoder.embedding(input_word)
                lstm_input = torch.cat([features, embedded], dim=1).unsqueeze(1)
                output, hidden = model.decoder.lstm(lstm_input, hidden)
                output = model.decoder.fc(output.squeeze(1))
                predicted = output.argmax(1).item()
                caption.append(predicted)
                
                if predicted == vocabulary.word_to_idx['<END>']:
                    break
            
            # Convert to words
            caption_words = []
            for idx in caption[1:]:
                word = vocabulary.idx_to_word[idx]
                if word == '<END>':
                    break
                if word not in ['<PAD>', '<START>']:
                    caption_words.append(word)
            
            predicted_captions.append(' '.join(caption_words))
    
    fig, axes = plt.subplots(1, num_images, figsize=(4*num_images, 4))
    if num_images == 1:
        axes = [axes]
    
    for i in range(num_images):
        image = images[i].cpu()
        mean = torch.tensor([0.485, 0.456, 0.406])
        std = torch.tensor([0.229, 0.224, 0.225])
        image = image * std.view(3, 1, 1) + mean.view(3, 1, 1)
        image = torch.clamp(image, 0, 1)
        
        # Convert numpy-transpose
        image_np = image.permute(1, 2, 0).numpy()
        axes[i].imshow(image_np)
        axes[i].axis('off')
        target_words = []
        for idx in target_captions[i]:
            if idx == 0:  
                break
            word = vocabulary.idx_to_word[idx.item()]
            if word not in ['<PAD>', '<START>', '<END>']:
                target_words.append(word)
        target_text = ' '.join(target_words)
    
        axes[i].set_title(f'Target: {target_text}\nPredicted: {predicted_captions[i]}', 
                         fontsize=8, wrap=True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"🖼️ Predictions visualization saved: {save_path}")
    
    plt.show()


def log_training_progress(epoch, train_loss, val_loss, train_acc, val_acc, 
                         bleu_score=None, log_file="training_log.txt"):
    """
    Log training progress to file
    
    Args:
        epoch (int): Current epoch
        train_loss (float): Training loss
        val_loss (float): Validation loss  
        train_acc (float): Training accuracy
        val_acc (float): Validation accuracy
        bleu_score (float): BLEU score (optional)
        log_file (str): Log file path
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    log_entry = f"[{timestamp}] Epoch {epoch:3d} | "
    log_entry += f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
    log_entry += f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}"
    
    if bleu_score is not None:
        log_entry += f" | BLEU: {bleu_score:.4f}"
    
    log_entry += "\n"
    
    # Writing to file
    with open(log_file, 'a') as f:
        f.write(log_entry)
    print(log_entry.strip())
