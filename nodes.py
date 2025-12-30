"""Main node definitions for WAN 2.2 BlockSwap functionality.

This module defines the main ComfyUI node class that users interact with.
It provides a clean interface to the block swapping functionality while
handling all the complex callback registration and parameter management.
"""

import comfy.model_management as mm
from comfy.patcher_extension import CallbacksMP
from comfy.model_patcher import ModelPatcher

from .config import BlockSwapConfig
from .callbacks import lazy_load_callback, cleanup_callback


class WanVideo22BlockSwap:
    """
    Block swapping for WAN 2.1/2.2 models with LAZY LOADING.

    Offloads transformer blocks to CPU DURING model loading to prevent
    VRAM spikes. Blocks are loaded directly to their target device
    instead of loading everything to GPU first.

    GGUF Compatible: Automatically detects and handles GGUF quantized models.

    This is the main ComfyUI node class that users will see in the UI.
    It provides all the configuration options and handles the callback
    registration for lazy loading and cleanup operations.
    """

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        """Get input types for the node."""
        return BlockSwapConfig.get_input_types()

    RETURN_TYPES: tuple = ("MODEL",)
    RETURN_NAMES: tuple = ("model",)
    CATEGORY: str = "ComfyUI-wan22Blockswap"
    FUNCTION: str = "apply_block_swap"
    DESCRIPTION: str = (
        "Apply LAZY LOADING block swapping to WAN 2.1/2.2 models. "
        "Blocks are offloaded DURING loading to prevent VRAM spikes. "
        "Apply block swapping to WAN 2.1/2.2 models to reduce VRAM usage. "
        "Swaps last N transformer blocks to CPU memory. "
        "Compatible with all WAN model variants (1.3B, 5B, 14B, LongCat). "
        "Supports optional VACE model block swapping for multi-modal tasks. "
        "GGUF compatible: Gracefully handles quantized models with best-effort swapping."
    )

    def apply_block_swap(
        self,
        model: ModelPatcher,
        blocks_to_swap: int,
        offload_txt_emb: bool,
        offload_img_emb: bool,
        use_non_blocking: bool = False,
        vace_blocks_to_swap: int = 0,
        prefetch_blocks: int = 0,
        block_swap_debug: bool = False,
    ) -> tuple:
        """
        Apply block swapping configuration to ComfyUI native WAN model.

        This function registers a callback that executes when the model is
        loaded, swapping the specified number of transformer blocks to the
        offload device (CPU) to reduce VRAM usage. The swap uses a clever
        strategy: keeps early blocks on GPU (where most computation happens)
        and moves later blocks to CPU (where activations can be staged).

        Args:
            model (ModelPatcher): The ComfyUI model patcher instance.
            blocks_to_swap (int): Number of transformer blocks to swap.
                Range depends on model variant:
                    - 1.3B/5B models: 0-30
                    - 14B model: 0-40
                    - LongCat: 0-48
            offload_txt_emb (bool): Whether to offload text embeddings to CPU.
            offload_img_emb (bool): Whether to offload image embeddings (I2V).
            use_non_blocking (bool): Use non-blocking transfers for speed.
            vace_blocks_to_swap (int): VACE blocks to swap (0=auto detection).
            prefetch_blocks (int): Blocks to prefetch ahead for pipeline.
            block_swap_debug (bool): Enable performance monitoring.

        Returns:
            tuple: Modified model patcher with block swap callback registered.

        Notes:
            - The callback is triggered on model load (CallbacksMP.ON_LOAD)
            - Block swap direction: LAST N blocks → CPU (not first N)
            - Early blocks remain on GPU where most computation occurs
            - Memory savings: ~100-200MB per 10 blocks swapped
            - Performance impact: varies with block prefetching settings
            - GGUF models: Best-effort swapping with graceful error handling

        Apply lazy loading block swapping - offloads DURING loading.

        Strategy:
        1. Hijack the model loading process
        2. As each block loads, immediately route to target device
        3. Prevents VRAM spike from full model load
        4. GPU only sees the blocks it needs to keep
        """

        def lazy_load_callback_wrapper(
            model_patcher: ModelPatcher,
            device_to,
            lowvram_model_memory,
            force_patch_weights,
            full_load,
        ) -> None:
            """Wrapper for lazy load callback with proper parameter passing."""
            lazy_load_callback(
                model_patcher=model_patcher,
                device_to=device_to,
                lowvram_model_memory=lowvram_model_memory,
                force_patch_weights=force_patch_weights,
                full_load=full_load,
                blocks_to_swap=blocks_to_swap,
                offload_txt_emb=offload_txt_emb,
                offload_img_emb=offload_img_emb,
                use_non_blocking=use_non_blocking,
                vace_blocks_to_swap=vace_blocks_to_swap,
                prefetch_blocks=prefetch_blocks,
                block_swap_debug=block_swap_debug,
            )

        # Clone model and register lazy loading callback
        model_copy = model.clone()

        # Register both ON_LOAD and ON_CLEANUP callbacks
        model_copy.add_callback(CallbacksMP.ON_LOAD, lazy_load_callback_wrapper)
        model_copy.add_callback(CallbacksMP.ON_CLEANUP, cleanup_callback)

        if block_swap_debug:
            print("[BlockSwap] Both ON_LOAD and ON_CLEANUP callbacks registered")

        return (model_copy,)


# Node registration
NODE_CLASS_MAPPINGS = {
    "wan22BlockSwap": WanVideo22BlockSwap,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "wan22BlockSwap": "WAN 2.2 BlockSwap (Lazy Load + GGUF Safe)",
}
