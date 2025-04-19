#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def create_stratified_splits(input_csv, output_dir, train_size=0.7, val_size=0.2, test_size=0.1, random_state=42):
    """
    Create stratified train-validation-test splits from a CSV file of image labels.
    
    Args:
        input_csv (str): Path to input CSV file with image paths and labels
        output_dir (str): Directory to save output CSV files
        train_size (float): Proportion of data to use for training (default: 0.7)
        val_size (float): Proportion of data to use for validation (default: 0.2)
        test_size (float): Proportion of data to use for testing (default: 0.1)
        random_state (int): Random seed for reproducibility
    
    Returns:
        tuple: Paths to the train, validation, and test CSV files
    """
    # Ensure the splits sum to 1
    assert abs(train_size + val_size + test_size - 1.0) < 1e-10, "Split sizes must sum to 1"
    
    # Read the input CSV file
    df = pd.read_csv(input_csv, header=None, names=['path', 'label'])
    
    # Display class distribution in the original dataset
    class_counts = df['label'].value_counts().sort_index()
    print(f"Original class distribution:")
    for label, count in class_counts.items():
        print(f"Class {label}: {count} images ({count/len(df)*100:.2f}%)")
    
    # First split: train + val vs test
    # Adjusted ratio for the first split to get the correct proportions
    temp_size = train_size + val_size
    first_test_size = test_size / temp_size
    
    train_val_df, test_df = train_test_split(
        df, 
        test_size=test_size,
        stratify=df['label'],
        random_state=random_state
    )
    
    # Second split: train vs val
    # Recalculate the validation size as a proportion of the train_val set
    second_val_size = val_size / (train_size + val_size)
    
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=second_val_size,
        stratify=train_val_df['label'],
        random_state=random_state
    )
    
    # Verify the splits
    print(f"\nSplit sizes:")
    print(f"Training set: {len(train_df)} images ({len(train_df)/len(df)*100:.2f}%)")
    print(f"Validation set: {len(val_df)} images ({len(val_df)/len(df)*100:.2f}%)")
    print(f"Test set: {len(test_df)} images ({len(test_df)/len(df)*100:.2f}%)")
    
    # Verify class distribution in each split
    print("\nClass distribution in each split:")
    for name, split_df in [("Training", train_df), ("Validation", val_df), ("Test", test_df)]:
        class_dist = split_df['label'].value_counts().sort_index()
        print(f"\n{name} set:")
        for label, count in class_dist.items():
            print(f"Class {label}: {count} images ({count/len(split_df)*100:.2f}%)")
    
    # Save the splits to CSV files
    train_csv = os.path.join(output_dir, 'aider_train.csv')
    val_csv = os.path.join(output_dir, 'aider_val.csv')
    test_csv = os.path.join(output_dir, 'aider_test.csv')
    
    train_df.to_csv(train_csv, index=False, header=False)
    val_df.to_csv(val_csv, index=False, header=False)
    test_df.to_csv(test_csv, index=False, header=False)
    
    print(f"\nSaved splits to:")
    print(f"Training set: {train_csv}")
    print(f"Validation set: {val_csv}")
    print(f"Test set: {test_csv}")
    
    return train_csv, val_csv, test_csv

def plot_class_distribution(input_csv, train_csv, val_csv, test_csv, output_dir):
    """
    Plot the class distribution in the original dataset and each split.
    
    Args:
        input_csv (str): Path to input CSV file
        train_csv (str): Path to training CSV file
        val_csv (str): Path to validation CSV file
        test_csv (str): Path to test CSV file
        output_dir (str): Directory to save the plot
    """
    # Read CSV files
    original_df = pd.read_csv(input_csv, header=None, names=['path', 'label'])
    train_df = pd.read_csv(train_csv, header=None, names=['path', 'label'])
    val_df = pd.read_csv(val_csv, header=None, names=['path', 'label'])
    test_df = pd.read_csv(test_csv, header=None, names=['path', 'label'])
    
    # Count classes
    original_counts = original_df['label'].value_counts().sort_index()
    train_counts = train_df['label'].value_counts().sort_index()
    val_counts = val_df['label'].value_counts().sort_index()
    test_counts = test_df['label'].value_counts().sort_index()
    
    # Plot
    plt.figure(figsize=(15, 8))
    x = np.arange(len(original_counts))
    width = 0.2
    
    plt.bar(x - 1.5*width, original_counts, width, label='Original')
    plt.bar(x - 0.5*width, train_counts, width, label='Train')
    plt.bar(x + 0.5*width, val_counts, width, label='Validation')
    plt.bar(x + 1.5*width, test_counts, width, label='Test')
    
    plt.xlabel('Class Label')
    plt.ylabel('Number of Images')
    plt.title('Class Distribution Across Splits')
    plt.xticks(x, original_counts.index)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Save the plot
    plot_path = os.path.join(output_dir, 'class_distribution.png')
    plt.savefig(plot_path)
    print(f"Saved class distribution plot to {plot_path}")
    
    # Calculate and plot percentages
    plt.figure(figsize=(15, 8))
    
    # Convert to percentages
    original_pct = original_counts / original_counts.sum() * 100
    train_pct = train_counts / train_counts.sum() * 100
    val_pct = val_counts / val_counts.sum() * 100
    test_pct = test_counts / test_counts.sum() * 100
    
    plt.bar(x - 1.5*width, original_pct, width, label='Original')
    plt.bar(x - 0.5*width, train_pct, width, label='Train')
    plt.bar(x + 0.5*width, val_pct, width, label='Validation')
    plt.bar(x + 1.5*width, test_pct, width, label='Test')
    
    plt.xlabel('Class Label')
    plt.ylabel('Percentage (%)')
    plt.title('Class Distribution Percentage Across Splits')
    plt.xticks(x, original_pct.index)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Save the percentage plot
    pct_plot_path = os.path.join(output_dir, 'class_distribution_percentage.png')
    plt.savefig(pct_plot_path)
    print(f"Saved percentage distribution plot to {pct_plot_path}")

if __name__ == "__main__":
    # Set paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_csv = os.path.join(base_dir, 'aider_labels.csv')
    output_dir = base_dir
    
    # Create the splits
    train_csv, val_csv, test_csv = create_stratified_splits(
        input_csv, 
        output_dir,
        train_size=0.7,
        val_size=0.2,
        test_size=0.1,
        random_state=42
    )
    
    # Plot the class distribution
    plot_class_distribution(input_csv, train_csv, val_csv, test_csv, output_dir)

