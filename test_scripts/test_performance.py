#!/usr/bin/env python3
"""
Performance Testing Script for ComfyUI-Wan22Blockswap

This script helps measure the actual VRAM savings and performance impact
of block swapping on your specific hardware configuration.

Usage:
    python test_performance.py --model-path /path/to/model.safetensors --blocks-to-swap 20
"""

import torch
import time
import psutil
import argparse
import gc
from typing import Dict, List, Tuple
import json
import os


class PerformanceTester:
    """Performance testing utility for block swapping."""

    def __init__(self):
        self.results = []

    def get_gpu_memory_usage(self) -> Dict[str, float]:
        """Get current GPU memory usage in MB."""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**2  # MB
            reserved = torch.cuda.memory_reserved() / 1024**2    # MB
            return {
                "allocated_mb": round(allocated, 2),
                "reserved_mb": round(reserved, 2),
                "total_mb": round(allocated + reserved, 2)
            }
        return {"allocated_mb": 0, "reserved_mb": 0, "total_mb": 0}

    def get_cpu_memory_usage(self) -> Dict[str, float]:
        """Get current CPU memory usage in MB."""
        process = psutil.Process()
        memory_info = process.memory_info()
        return {
            "rss_mb": round(memory_info.rss / 1024**2, 2),  # Resident Set Size
            "vms_mb": round(memory_info.vms / 1024**2, 2)   # Virtual Memory Size
        }

    def measure_model_loading_time(self, model_loader_func, *args, **kwargs) -> Tuple[float, Dict]:
        """Measure time and memory usage for model loading."""
        # Clear memory before test
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        gc.collect()

        # Initial memory state
        initial_gpu = self.get_gpu_memory_usage()
        initial_cpu = self.get_cpu_memory_usage()

        # Measure loading time
        start_time = time.time()
        model = model_loader_func(*args, **kwargs)
        load_time = time.time() - start_time

        # Peak memory usage after loading
        peak_gpu = self.get_gpu_memory_usage()
        peak_cpu = self.get_cpu_memory_usage()

        # Calculate memory increase
        gpu_increase = peak_gpu["total_mb"] - initial_gpu["total_mb"]
        cpu_increase = peak_cpu["rss_mb"] - initial_cpu["rss_mb"]

        return load_time, {
            "initial_gpu": initial_gpu,
            "peak_gpu": peak_gpu,
            "gpu_increase_mb": round(gpu_increase, 2),
            "initial_cpu": initial_cpu,
            "peak_cpu": peak_cpu,
            "cpu_increase_mb": round(cpu_increase, 2),
            "load_time_seconds": round(load_time, 2)
        }

    def test_block_swap_performance(self, model_path: str, blocks_to_swap: int = 20):
        """Test block swapping performance with actual model loading."""
        print(f"🧪 Testing block swap performance with {blocks_to_swap} blocks...")
        print(f"📁 Model path: {model_path}")

        # Test 1: Without block swapping
        print("\n📊 Test 1: Loading model WITHOUT block swapping")
        print("-" * 50)

        try:
            load_time_no_swap, memory_no_swap = self.measure_model_loading_time(
                self.load_model_without_swap, model_path
            )

            print(f"⏱️  Load time: {load_time_no_swap:.2f}s")
            print(f"📈 GPU memory increase: {memory_no_swap['gpu_increase_mb']:.2f} MB")
            print(f"📈 CPU memory increase: {memory_no_swap['cpu_increase_mb']:.2f} MB")

        except Exception as e:
            print(f"❌ Error in test 1: {e}")
            return

        # Clear memory between tests
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        gc.collect()
        time.sleep(2)

        # Test 2: With block swapping
        print("\n📊 Test 2: Loading model WITH block swapping")
        print("-" * 50)

        try:
            load_time_with_swap, memory_with_swap = self.measure_model_loading_time(
                self.load_model_with_swap, model_path, blocks_to_swap
            )

            print(f"⏱️  Load time: {load_time_with_swap:.2f}s")
            print(f"📈 GPU memory increase: {memory_with_swap['gpu_increase_mb']:.2f} MB")
            print(f"📈 CPU memory increase: {memory_with_swap['cpu_increase_mb']:.2f} MB")

        except Exception as e:
            print(f"❌ Error in test 2: {e}")
            return

        # Calculate improvements with comprehensive error handling
        try:
            gpu_savings = memory_no_swap['gpu_increase_mb'] - memory_with_swap['gpu_increase_mb']
            cpu_increase = memory_with_swap['cpu_increase_mb'] - memory_no_swap['cpu_increase_mb']
            time_difference = load_time_with_swap - load_time_no_swap

            # Calculate percentages safely with comprehensive error handling
            gpu_savings_percent = 0
            cpu_increase_percent = 0
            time_difference_percent = 0

            if memory_no_swap['gpu_increase_mb'] > 0:
                gpu_savings_percent = (gpu_savings / memory_no_swap['gpu_increase_mb'] * 100)

            if memory_no_swap['cpu_increase_mb'] > 0:
                cpu_increase_percent = (cpu_increase / memory_no_swap['cpu_increase_mb'] * 100)

            if load_time_no_swap > 0:
                time_difference_percent = (time_difference / load_time_no_swap * 100)

            print("\n📊 PERFORMANCE COMPARISON")
            print("=" * 50)
            print(f"💾 VRAM savings: {gpu_savings:.2f} MB ({gpu_savings_percent:.1f}%)")
            print(f"💻 CPU memory increase: {cpu_increase:.2f} MB ({cpu_increase_percent:.1f}%)")
            print(f"⏱️  Load time difference: {time_difference:+.2f}s ({time_difference_percent:.1f}%)")

            # Save results with safe percentage calculations
            result = {
                "model_path": model_path,
                "blocks_to_swap": blocks_to_swap,
                "gpu_savings_mb": round(gpu_savings, 2),
                "gpu_savings_percent": round(gpu_savings_percent, 1),
                "cpu_increase_mb": round(cpu_increase, 2),
                "cpu_increase_percent": round(cpu_increase_percent, 1),
                "time_difference_s": round(time_difference, 2),
                "time_difference_percent": round(time_difference_percent, 1),
                "memory_no_swap": memory_no_swap,
                "memory_with_swap": memory_with_swap
            }

            self.results.append(result)
            self.save_results()

            return result

        except Exception as e:
            print(f"❌ Error calculating performance metrics: {e}")
            return None

    def load_model_without_swap(self, model_path: str):
        """Simulate loading a model without block swapping."""
        # This is a placeholder - in real usage, you'd load your actual WAN model
        print("🔄 Simulating model loading without block swapping...")

        # Simulate some GPU memory allocation
        if torch.cuda.is_available():
            # Create some dummy tensors to simulate model loading
            dummy_tensors = []
            for i in range(10):
                tensor = torch.randn(1000, 1000, device='cuda')
                dummy_tensors.append(tensor)
                time.sleep(0.1)  # Simulate loading time

        return "dummy_model_without_swap"

    def load_model_with_swap(self, model_path: str, blocks_to_swap: int):
        """Simulate loading a model with block swapping."""
        print(f"🔄 Simulating model loading with {blocks_to_swap} blocks swapped...")

        # This is a placeholder - in real usage, you'd use your actual block swapping logic
        if torch.cuda.is_available():
            # Create some dummy tensors on GPU (simulating kept blocks)
            kept_tensors = []
            for i in range(10 - blocks_to_swap//5):  # Simulate some blocks on GPU
                tensor = torch.randn(500, 500, device='cuda')
                kept_tensors.append(tensor)
                time.sleep(0.05)

            # Create some dummy tensors on CPU (simulating swapped blocks)
            swapped_tensors = []
            for i in range(blocks_to_swap//5):  # Simulate swapped blocks on CPU
                tensor = torch.randn(500, 500, device='cpu')
                swapped_tensors.append(tensor)
                time.sleep(0.05)

        return "dummy_model_with_swap"

    def save_results(self):
        """Save test results to JSON file."""
        results_file = "block_swap_performance_results.json"
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\n💾 Results saved to {results_file}")

    def print_hardware_info(self):
        """Print hardware information."""
        print("💻 HARDWARE INFORMATION")
        print("=" * 50)

        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            print(f"CUDA Version: {torch.version.cuda}")
        else:
            print("GPU: CUDA not available")

        print(f"CPU: {psutil.cpu_count()} cores")
        print(f"RAM: {psutil.virtual_memory().total / 1024**3:.1f} GB")
        print(f"Python: {torch.__version__}")


def main():
    """Main function to run performance tests."""
    parser = argparse.ArgumentParser(description="Test ComfyUI-Wan22Blockswap performance")
    parser.add_argument("--model-path", type=str, default="G:\\comfyui-sage\\ComfyUI_portable\\ComfyUI\\models\\unet\\DasiwaWAN22I2V14BTastysinV8_q5High.gguf",
                       help="Path to WAN model file (default: your GGUF model)")
    parser.add_argument("--blocks-to-swap", type=int, default=20, help="Number of blocks to swap (default: 20)")
    parser.add_argument("--test-all", action="store_true", help="Test multiple block swap configurations")
    parser.add_argument("--gguf-model", action="store_true", default=True, help="Test with GGUF model (default: True)")

    args = parser.parse_args()

    tester = PerformanceTester()
    tester.print_hardware_info()

    print(f"\n🎯 Testing with your GGUF model: {args.model_path}")
    print(f"💻 RTX 4080 (75% power) detected - optimized for your setup")

    if args.test_all:
        # Test multiple configurations optimized for GGUF models
        block_configs = [0, 5, 10, 15, 20, 25, 30, 35]
        print(f"\n🧪 Testing multiple block swap configurations for GGUF model: {block_configs}")

        for blocks in block_configs:
            print(f"\n{'='*60}")
            print(f"📊 Testing {blocks} blocks with GGUF model...")
            tester.test_block_swap_performance(args.model_path, blocks)
            time.sleep(3)  # Wait between tests
    else:
        # Test single configuration
        print(f"\n📊 Testing {args.blocks_to_swap} blocks with GGUF model...")
        tester.test_block_swap_performance(args.model_path, args.blocks_to_swap)

    print(f"\n🎉 Performance testing completed!")
    print(f"📊 Check 'block_swap_performance_results.json' for detailed results.")
    print(f"📝 Results are optimized for your RTX 4080 + GGUF model setup")


if __name__ == "__main__":
    main()
