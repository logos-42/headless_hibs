"""
Main entry point for Twistor-inspired Liquid Neural Network training.

This script demonstrates training the Twistor LMT on a sine wave prediction task.
"""

import torch
import yaml
import os
from liquid_net import train_twistor_LMT


def load_config(config_path: str = "liquid_net/configs/config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    """Main training function."""
    # Load configuration
    config = load_config()
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Train model
    model, history = train_twistor_LMT(
        n_epochs=config['training']['n_epochs'],
        batch_size=config['training']['batch_size'],
        lr=config['training']['lr'],
        hidden_dim=config['model']['hidden_dim'],
        stability_weight=config['training']['stability_weight'],
        device=device
    )
    
    # Save model
    if config['logging']['save_model']:
        torch.save(model.state_dict(), config['logging']['model_path'])
        print(f"Model saved to '{config['logging']['model_path']}'")
    
    # Print summary
    print()
    print("=" * 60)
    print("Training Summary:")
    print(f"  Initial Train Loss: {history['train_loss'][0]:.6f}")
    print(f"  Final Train Loss: {history['train_loss'][-1]:.6f}")
    print(f"  Initial Val MSE: {history['val_mse'][0]:.6f}")
    print(f"  Final Val MSE: {history['val_mse'][-1]:.6f}")
    print(f"  Convergence: {'Yes' if history['train_loss'][-1] < history['train_loss'][0] * 0.5 else 'Partial'}")
    print("=" * 60)


if __name__ == '__main__':
    main()
