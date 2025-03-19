import subprocess

from utils.init_models import get_backend_device


def get_gpu_info() -> str:
    gpu_info = subprocess.check_output(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader,nounits"])
    gpu_model = gpu_info.decode("utf-8").strip().split("\n")[-1]
    return gpu_model.lower()

def get_optimised_hyperparameters() -> dict:
    device = get_backend_device()
    if device != 'cuda':
        return hyperparameters['m2']
    
    gpu_model = get_gpu_info()
    assert gpu_model in hyperparameters, f"Unsupported GPU model: {gpu_model}"
    return hyperparameters[gpu_model]


hyperparameters = {
    'm2': {
        'sample_size': 0.1,
        'per_device_train_batch_size': 2,
        'per_device_eval_batch_size': 2,
        'gradient_accumulation_steps': 2,
        'learning_rate': 3e-5,
        'num_train_epochs': 3,
        'warmup_steps': 50,
        'weight_decay': 0.01,
        'fp16': False,
        'device': 'mps',
        'save_steps': 250,          # Smaller dataset, save every 250 steps ( ~1/4 epoch)
        'save_total_limit': 2,      # Keep 2 checkpoints to save space
        'eval_strategy': "epoch",   # Eval per epoch (3 total)
        'save_strategy': "epoch",   # Save per epoch (matches eval)
        'logging_steps': 50,         # Log frequently due to small dataset
    },
    't4': {
        'sample_size': 1.0,
        'per_device_train_batch_size': 8,
        'per_device_eval_batch_size': 8,
        'gradient_accumulation_steps': 2,
        'learning_rate': 3e-5,
        'num_train_epochs': 5,
        'warmup_steps': 100,
        'weight_decay': 0.01,
        'fp16': True,
        'device': 'cuda',
        'save_steps': 500,          # Save every 500 steps (~1/4 epoch with effective batch 16)
        'save_total_limit': 2,      # Keep 2 checkpoints
        'eval_strategy': "epoch",   # Eval per epoch
        'save_strategy': "epoch",   # Save per epoch
        'logging_steps': 100,        # Log every 100 steps for decent granularity
    },
    'a10g': {
        'sample_size': 1.0,
        'per_device_train_batch_size': 16,
        'per_device_eval_batch_size': 16,
        'gradient_accumulation_steps': 1,
        'learning_rate': 3e-5,
        'num_train_epochs': 5,
        'warmup_steps': 100,
        'weight_decay': 0.01,
        'fp16': True,
        'device': 'cuda',
        'save_steps': 500,          # Save every 500 steps (~1/2 epoch with batch 16)
        'save_total_limit': 2,      # Keep 2 checkpoints
        'eval_strategy': "epoch",   # Eval per epoch
        'save_strategy': "epoch",   # Save per epoch
        'logging_steps': 100,        # Log every 100 steps for monitoring
    },
}
