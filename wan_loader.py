"""
WANModelLoader: A simple, clean all-in-one WAN model loader.

This loader handles both safetensors and GGUF WAN models with
automatic version and variant detection. No BlockSwap complexity -
just load the model and let the separate BlockSwap nodes handle
block swapping if needed.

Supports:
- WAN 2.1 (1.3B, 5B, 14B) and WAN 2.2 models
- All variants: T2V, I2V, VACE, Camera, S2V, Humo, Animate
- safetensors and GGUF formats
- Optional FP8 quantization for additional VRAM savings
"""

import torch
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

import folder_paths
import comfy.model_management as mm
import comfy.utils

from .loader_helpers import (
    detect_model_format,
    load_state_dict_to_cpu,
    create_model_skeleton,
    create_model_patcher,
)
from .model_detection import detect_wan_config

logger = logging.getLogger("WANModelLoader")


def _register_gguf_extension():
    """Add GGUF extension support to diffusion_models folder type."""
    if "diffusion_models" in folder_paths.folder_names_and_paths:
        paths, extensions = folder_paths.folder_names_and_paths["diffusion_models"]
        if ".gguf" not in extensions:
            extensions.add(".gguf")
            folder_paths.folder_names_and_paths["diffusion_models"] = (paths, extensions)
            if hasattr(folder_paths, "filename_list_cache"):
                if "diffusion_models" in folder_paths.filename_list_cache:
                    del folder_paths.filename_list_cache["diffusion_models"]
            logger.info("Registered .gguf extension for diffusion_models folder")

_register_gguf_extension()


class WANModelLoader:
    """
    Simple all-in-one WAN model loader.

    Loads WAN 2.1/2.2 models in safetensors or GGUF format with
    automatic configuration detection. No BlockSwap - just pure
    model loading.

    For BlockSwap functionality, connect the output to a
    WAN22BlockSwap node.

    Features:
    - Auto-detects WAN version (2.1/2.2) and variant (T2V/I2V/etc)
    - Supports safetensors and GGUF quantized models
    - Optional FP8 optimization for additional VRAM savings
    - Clean, simple interface
    """

    CATEGORY = "WAN"
    FUNCTION = "load_model"
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        """Define node inputs."""
        try:
            all_models = folder_paths.get_filename_list("diffusion_models")
        except Exception:
            all_models = []

        # Filter by format
        safetensors_ext = [".safetensors", ".sft"]
        safetensors_models = sorted([
            p for p in all_models
            if any(p.lower().endswith(ext) for ext in safetensors_ext)
        ])

        gguf_models = sorted([
            p for p in all_models
            if p.lower().endswith(".gguf")
        ])

        if not safetensors_models:
            safetensors_models = ["no safetensors models found"]
        if not gguf_models:
            gguf_models = ["no gguf models found"]

        return {
            "required": {
                "model_type": (["safetensors", "gguf"], {
                    "default": "safetensors",
                    "tooltip": "Model format to load."
                }),
                "safetensors_model": (safetensors_models, {
                    "tooltip": "Safetensors model from diffusion_models folder"
                }),
                "gguf_model": (gguf_models, {
                    "tooltip": "GGUF quantized model from diffusion_models folder"
                }),
            },
            "optional": {
                "wan_version": (["auto", "2.1", "2.2"], {
                    "default": "auto",
                    "tooltip": "WAN model version. 'auto' detects from weights."
                }),
                "model_variant": (
                    ["auto", "t2v", "i2v", "vace", "camera", "s2v", "humo", "animate"],
                    {
                        "default": "auto",
                        "tooltip": "WAN model variant. 'auto' detects from weights."
                    }
                ),
                "fp8_optimization": (["disabled", "e4m3fn", "e5m2"], {
                    "default": "disabled",
                    "tooltip": "Apply FP8 quantization for additional memory savings."
                }),
                "weight_dtype": (["auto", "fp16", "bf16", "fp32"], {
                    "default": "auto",
                    "tooltip": "Data type for model weights. 'auto' uses native dtype."
                }),
            }
        }

    def load_model(
        self,
        model_type: str,
        safetensors_model: str,
        gguf_model: str,
        wan_version: str = "auto",
        model_variant: str = "auto",
        fp8_optimization: str = "disabled",
        weight_dtype: str = "auto",
    ) -> Tuple[Any]:
        """
        Load a WAN model.

        Args:
            model_type: "safetensors" or "gguf"
            safetensors_model: Model name for safetensors format
            gguf_model: Model name for GGUF format
            wan_version: WAN version ("auto", "2.1", "2.2")
            model_variant: Model variant type
            fp8_optimization: FP8 quantization mode
            weight_dtype: Weight data type

        Returns:
            Tuple containing ModelPatcher
        """
        # Determine model path
        if model_type == "gguf":
            model_path = gguf_model
            if model_path == "no gguf models found":
                raise FileNotFoundError(
                    "No GGUF models found. Add .gguf files to models/diffusion_models/"
                )
        else:
            model_path = safetensors_model
            if model_path == "no safetensors models found":
                raise FileNotFoundError(
                    "No safetensors models found. Add .safetensors files to models/diffusion_models/"
                )

        full_path = folder_paths.get_full_path("diffusion_models", model_path)
        if not full_path:
            raise FileNotFoundError(f"Model not found: {model_path}")

        logger.info("╔══════════════════════════════════════════════════════════════")
        logger.info(f"║ WAN Model Loader")
        logger.info(f"║ Model: {model_path}")
        logger.info(f"║ Format: {model_type}")
        logger.info("╚══════════════════════════════════════════════════════════════")

        # Memory cleanup
        mm.unload_all_models()
        mm.soft_empty_cache()

        # Load state dict
        model_format = detect_model_format(full_path)
        logger.info(f"Loading {model_format} model to CPU...")

        state_dict, _, metadata = load_state_dict_to_cpu(full_path, model_format)
        logger.info(f"Loaded {len(state_dict)} tensors")

        # Detect GGUF
        is_gguf_model = False
        for tensor in state_dict.values():
            if hasattr(tensor, 'tensor_type') and tensor.tensor_type is not None:
                is_gguf_model = True
                break

        # Detect WAN config
        logger.info("Detecting WAN configuration...")
        try:
            wan_config = detect_wan_config(
                state_dict,
                wan_version=wan_version,
                model_variant=model_variant,
            )
        except ValueError as e:
            raise ValueError(f"Not a valid WAN model: {e}")

        logger.info(
            f"Detected: WAN {wan_config['wan_version']}, "
            f"variant={wan_config['model_variant']}, "
            f"size={wan_config.get('model_size', 'unknown')}, "
            f"blocks={wan_config['num_layers']}, "
            f"dim={wan_config['dim']}"
        )

        # Determine dtype
        if weight_dtype == "auto":
            sample_tensor = next(iter(state_dict.values()))
            is_ggml = hasattr(sample_tensor, 'tensor_type') and sample_tensor.tensor_type is not None

            if is_ggml:
                model_dtype = torch.float16
            elif sample_tensor.dtype in [torch.float16, torch.bfloat16, torch.float32]:
                model_dtype = sample_tensor.dtype
            else:
                model_dtype = torch.float16
        else:
            dtype_map = {
                "fp16": torch.float16,
                "bf16": torch.bfloat16,
                "fp32": torch.float32,
            }
            model_dtype = dtype_map.get(weight_dtype, torch.float16)

        logger.info(f"Using dtype: {model_dtype}")

        # Create model
        logger.info("Creating model...")
        model = create_model_skeleton(
            wan_config=wan_config,
            dtype=model_dtype,
            device="meta",
            is_gguf=is_gguf_model,
        )

        # Assign weights - all to GPU (no block routing here)
        main_device = mm.get_torch_device()
        logger.info(f"Assigning weights to {main_device}...")

        self._assign_weights_simple(
            model=model,
            state_dict=state_dict,
            device=main_device,
            fp8_optimization=fp8_optimization,
        )

        # Create patcher
        model_patcher = create_model_patcher(
            diffusion_model=model,
            wan_config=wan_config,
            load_device=main_device,
            offload_device=mm.unet_offload_device(),
            blocks_to_swap=0,  # No pre-routing
            is_gguf=is_gguf_model,
        )

        # Cleanup
        del state_dict
        mm.soft_empty_cache()

        logger.info("╔══════════════════════════════════════════════════════════════")
        logger.info(f"║ ✓ Model loaded successfully!")
        logger.info(f"║   Use WAN22BlockSwap node for block swapping if needed")
        logger.info("╚══════════════════════════════════════════════════════════════")

        return (model_patcher,)

    def _assign_weights_simple(
        self,
        model: Any,
        state_dict: Dict[str, torch.Tensor],
        device: torch.device,
        fp8_optimization: str = "disabled",
    ) -> None:
        """
        Assign weights from state dict to model.

        Simple weight assignment without block routing.
        All weights go to the specified device.

        Args:
            model: The model to assign weights to
            state_dict: State dict with weights
            device: Target device for all weights
            fp8_optimization: FP8 quantization mode
        """
        # Get FP8 dtype if enabled
        fp8_dtype = None
        if fp8_optimization == "e4m3fn" and hasattr(torch, 'float8_e4m3fn'):
            fp8_dtype = torch.float8_e4m3fn
        elif fp8_optimization == "e5m2" and hasattr(torch, 'float8_e5m2'):
            fp8_dtype = torch.float8_e5m2

        assigned = 0
        skipped = 0

        for key, tensor in state_dict.items():
            # Find parameter in model
            parts = key.split('.')
            target = model

            try:
                for part in parts[:-1]:
                    if part.isdigit():
                        target = target[int(part)]
                    else:
                        target = getattr(target, part)

                param_name = parts[-1]

                # Check if it's a GGMLTensor (GGUF quantized)
                is_ggml = hasattr(tensor, 'tensor_type') and tensor.tensor_type is not None

                if is_ggml:
                    # GGMLTensor - don't move, let ComfyUI-GGUF handle it
                    if hasattr(target, param_name):
                        setattr(target, param_name, tensor)
                        assigned += 1
                else:
                    # Regular tensor - move to device
                    tensor_device = tensor.to(device)

                    # Apply FP8 if enabled
                    if fp8_dtype is not None and tensor_device.dtype in [torch.float16, torch.bfloat16, torch.float32]:
                        tensor_device = tensor_device.to(fp8_dtype)

                    if hasattr(target, param_name):
                        param = getattr(target, param_name)
                        if isinstance(param, torch.nn.Parameter):
                            param.data = tensor_device
                        else:
                            setattr(target, param_name, tensor_device)
                        assigned += 1
                    else:
                        skipped += 1

            except (AttributeError, IndexError, KeyError):
                skipped += 1
                continue

        logger.info(f"Assigned {assigned} weights, skipped {skipped}")


# Node registration
NODE_CLASS_MAPPINGS = {
    "WANModelLoader": WANModelLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WANModelLoader": "WAN Model Loader",
}
