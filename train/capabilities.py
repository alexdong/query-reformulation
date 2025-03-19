from typing import Any, Dict

import torch


def show_capabilities() -> Dict[str, Any]:
    """
    Interrogates the PyTorch library to determine the available hardware and software capabilities.
    Returns a dictionary with information about available hardware and PyTorch configuration.

      if capabilities["cuda_available"]:
        # CUDA is available
      elif capabilities["mps_available"]:
        # Apple Silicon GPU detected
      else:
        # No GPU detected, using CPU only

    """
    capabilities = {}
    
    # PyTorch version
    capabilities["torch_version"] = torch.__version__
    
    # CUDA availability and version
    capabilities["cuda_available"] = torch.cuda.is_available()
    if capabilities["cuda_available"]:
        capabilities["cuda_version"] = torch.version.cuda
        capabilities["cuda_device_count"] = torch.cuda.device_count()
        capabilities["cuda_device_names"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        capabilities["cuda_current_device"] = torch.cuda.current_device()
        capabilities["cuda_arch_list"] = torch.cuda.get_arch_list() if hasattr(torch.cuda, 'get_arch_list') else "Not available"
    
    # MPS (Metal Performance Shaders) for Mac
    if hasattr(torch, 'mps'):
        capabilities["mps_available"] = torch.backends.mps.is_available()
    else:
        capabilities["mps_available"] = False
    
    # CPU capabilities
    capabilities["num_threads"] = torch.get_num_threads()
    capabilities["num_interop_threads"] = torch.get_num_interop_threads()
    
    # Check for various backends
    capabilities["backends"] = {}
    
    # CUDA backend details
    if hasattr(torch.backends, 'cuda'):
        capabilities["backends"]["cuda"] = {
            "is_built": torch.backends.cuda.is_built(),
        }
        if torch.backends.cuda.is_built():
            capabilities["backends"]["cuda"].update({
                "cudnn_enabled": torch.backends.cudnn.enabled,
                "cudnn_version": torch.backends.cudnn.version() if hasattr(torch.backends.cudnn, 'version') else "Not available",
                "cudnn_deterministic": torch.backends.cudnn.deterministic,
                "cudnn_benchmark": torch.backends.cudnn.benchmark,
            })
    
    # MKL backend
    if hasattr(torch.backends, 'mkl'):
        capabilities["backends"]["mkl"] = {
            "is_available": torch.backends.mkl.is_available(),
            "version": torch.backends.mkl.get_version_string() if hasattr(torch.backends.mkl, 'get_version_string') else "Not available",
        }
    
    # OpenMP backend
    if hasattr(torch.backends, 'openmp'):
        capabilities["backends"]["openmp"] = {
            "is_available": torch.backends.openmp.is_available(),
        }
    
    # ONNX runtime
    try:
        import onnxruntime as ort
        capabilities["onnx"] = {
            "available": True,
            "version": ort.__version__,
            "providers": ort.get_available_providers(),
        }
    except ImportError:
        capabilities["onnx"] = {"available": False}
    
    # Print capabilities in a readable format
    print(f"[INFO] PyTorch version: {capabilities['torch_version']}")
    
    print("\n[INFO] CUDA Information:")
    if capabilities["cuda_available"]:
        print("  - CUDA available: Yes")
        print(f"  - CUDA version: {capabilities['cuda_version']}")
        print(f"  - CUDA device count: {capabilities['cuda_device_count']}")
        for i, device in enumerate(capabilities['cuda_device_names']):
            print(f"  - CUDA device {i}: {device}")
        print(f"  - CUDA architectures: {capabilities['cuda_arch_list']}")
    else:
        print("  - CUDA available: No")
    
    print("\n[INFO] MPS Information:")
    print(f"  - MPS available: {capabilities['mps_available']}")
    
    print("\n[INFO] CPU Information:")
    print(f"  - Number of threads: {capabilities['num_threads']}")
    print(f"  - Number of interop threads: {capabilities['num_interop_threads']}")
    
    print("\n[INFO] Backend Information:")
    for backend, info in capabilities.get("backends", {}).items():
        print(f"  - {backend.upper()}:")
        for key, value in info.items():
            print(f"    - {key}: {value}")
    
    print("\n[INFO] ONNX Runtime:")
    if capabilities["onnx"]["available"]:
        print("  - Available: Yes")
        print(f"  - Version: {capabilities['onnx']['version']}")
        print(f"  - Providers: {', '.join(capabilities['onnx']['providers'])}")
    else:
        print("  - Available: No")
    
    return capabilities


if __name__ == "__main__":
    print("[LEVEL] Checking PyTorch capabilities...")
    capabilities = show_capabilities()
    
    # Recommend model size based on hardware
    if capabilities["cuda_available"] and capabilities["cuda_device_count"] > 0:
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # Convert to GB
        print(f"\n[LEVEL] GPU memory: {gpu_mem:.2f} GB")
        
        if gpu_mem > 24:
            print("[LEVEL] Hardware supports large models (24GB+ VRAM)")
            print("[LEVEL] Recommended: Use 'large' model variants")
        elif gpu_mem > 12:
            print("[LEVEL] Hardware supports medium-sized models (12GB+ VRAM)")
            print("[LEVEL] Recommended: Use 'base' model variants")
        elif gpu_mem > 6:
            print("[LEVEL] Hardware supports small models (6GB+ VRAM)")
            print("[LEVEL] Recommended: Use quantized models")
        else:
            print("[LEVEL] Limited GPU memory (<6GB VRAM)")
            print("[LEVEL] Recommended: Use CPU or highly quantized models")
    elif capabilities["mps_available"]:
        print("\n[LEVEL] Apple Silicon GPU detected")
        print("[LEVEL] Recommended: Use 'base' model variants with MPS acceleration")
    else:
        print("\n[LEVEL] No GPU detected, using CPU only")
        print("[LEVEL] Recommended: Use quantized models or ONNX runtime for faster inference")

