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
        'per_device_eval_batch_size': 1,
        'gradient_accumulation_steps': 2,
        'learning_rate': 3e-5,
        'num_train_epochs': 3,
        'warmup_steps': 50,
        'weight_decay': 0.01,
        'fp16': False,
        'device': 'mps',
        'save_strategy': "epoch",
        'save_steps': 250,
        'save_total_limit': 2,
        'eval_strategy': "steps",
        'eval_steps': 250,
        'logging_steps': 50,
    },
    'tesla t4': {
        'sample_size': 1,
        'per_device_train_batch_size': 8, # Can go higher but throughput max out at 8
        'per_device_eval_batch_size': 4,
        'gradient_accumulation_steps': 4,
        'learning_rate': 3e-5,
        'num_train_epochs': 5,
        'warmup_steps': 100,
        'weight_decay': 0.01,
        'fp16': False,
        'device': 'cuda',
        'save_strategy': "epoch",
        'save_steps': 500,
        'save_total_limit': 2,
        'eval_strategy': "epoch",
        'eval_steps': 100,
        'logging_steps': 100,
    },
    'nvidia a10g': {
        'sample_size': 1.0,
        'per_device_train_batch_size': 16,
        'per_device_eval_batch_size': 1,
        'gradient_accumulation_steps': 1,
        'learning_rate': 3e-5,
        'num_train_epochs': 10,
        'warmup_steps': 100,
        'weight_decay': 0.01,
        'fp16': False,
        'device': 'cuda',
        'save_strategy': "epoch",
        'save_steps': 500,
        'save_total_limit': 2,
        'eval_strategy': "epoch",
        'eval_steps': 500,
        'logging_steps': 100,
    },
}
