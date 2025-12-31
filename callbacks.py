"""Callback functions for WAN 2.2 BlockSwap operations.

This module contains the callback functions that are executed during
model loading and cleanup. These callbacks implement the lazy loading
strategy and proper cleanup of swapped blocks.

The cleanup_callback integrates with BlockSwapModelTracker to make smart
cleanup decisions based on session state, preventing cleanup from deleting
blocks that subsequent loops still need.
"""

import torch
import gc
from typing import Any, Optional, Dict
from tqdm import tqdm

from .block_manager import BlockManager, BlockSwapTracker
from .utils import log_debug, sync_gpu, clear_device_caches
from .model_tracker import BlockSwapModelTracker, CleanupDecision


def lazy_load_callback(
    model_patcher: Any,
    device_to: torch.device,
    lowvram_model_memory: int,
    force_patch_weights: bool,
    full_load: bool,
    blocks_to_swap: int,
    offload_txt_emb: bool,
    offload_img_emb: bool,
    use_non_blocking: bool,
    vace_blocks_to_swap: int,
    prefetch_blocks: int,
    block_swap_debug: bool,
) -> None:
    """
    LAZY LOADING: Load blocks directly to target device.

    This prevents the massive VRAM spike from loading the entire
    model to GPU first. Blocks that should be on CPU are loaded
    directly to CPU, never touching GPU VRAM.

    Args:
        model_patcher: The model patcher instance
        device_to: Target device for model loading
        lowvram_model_memory: Memory limit for low VRAM mode
        force_patch_weights: Whether to force weight patching
        full_load: Whether to perform full model loading
        blocks_to_swap: Number of blocks to swap to CPU
        offload_txt_emb: Whether to offload text embeddings
        offload_img_emb: Whether to offload image embeddings
        use_non_blocking: Whether to use non-blocking transfers
        vace_blocks_to_swap: Number of VACE blocks to swap
        prefetch_blocks: Number of blocks to prefetch
        block_swap_debug: Enable debug logging
    """
    # Initialize block tracking for ON_CLEANUP
    blockswap_tracking = BlockSwapTracker()
    blockswap_tracking.blocks_to_swap = blocks_to_swap

    base_model = model_patcher.model
    main_device = torch.device("cuda")
    offload_device = model_patcher.offload_device

    # Detect GGUF model
    model_type_str = str(type(base_model).__module__)
    blockswap_tracking.is_gguf_model = BlockManager.detect_gguf_model(model_type_str)

    if block_swap_debug:
        if blockswap_tracking.is_gguf_model:
            print("[BlockSwap] GGUF model detected - using LAZY LOADING with safe swapping")
        else:
            print("[BlockSwap] Using LAZY LOADING - blocks loaded directly to target device")
        print("[BlockSwap] This prevents VRAM spikes during loading!")

    # Get UNet model
    unet = BlockManager.get_unet_from_model(base_model)
    if unet is None or not hasattr(unet, "blocks"):
        return

    # Calculate swap indices
    swap_start_idx, actual_blocks_to_swap = BlockManager.calculate_swap_indices(
        len(unet.blocks), blocks_to_swap
    )

    if block_swap_debug:
        print(f"[BlockSwap] Total blocks: {len(unet.blocks)}")
        print(f"[BlockSwap] Keeping on GPU: blocks 0-{swap_start_idx-1}")
        print(f"[BlockSwap] Offloading to CPU: blocks {swap_start_idx}-{len(unet.blocks)-1}")

    # Perform block swapping
    gpu_blocks, cpu_blocks, failed_swaps = BlockManager.swap_transformer_blocks(
        unet, swap_start_idx, len(unet.blocks), offload_device, main_device,
        use_non_blocking, block_swap_debug, blockswap_tracking
    )

    # Offload embeddings
    BlockManager.offload_embeddings(
        unet, offload_device, use_non_blocking, offload_txt_emb, offload_img_emb,
        block_swap_debug, blockswap_tracking
    )

    # Handle VACE blocks
    if vace_blocks_to_swap > 0:
        vace_success = BlockManager.handle_vace_blocks(
            unet, vace_blocks_to_swap, offload_device, main_device,
            use_non_blocking, block_swap_debug
        )

    # Final cleanup
    BlockManager._clear_gpu_cache()

    if block_swap_debug:
        print(f"[BlockSwap] ===== LAZY LOADING COMPLETE =====")
        print(f"[BlockSwap] GPU blocks: {gpu_blocks}")
        print(f"[BlockSwap] CPU blocks: {cpu_blocks}")
        if failed_swaps > 0:
            print(f"[BlockSwap] Failed/Skipped: {failed_swaps} (GGUF quantized)")
        print(f"[BlockSwap] Peak VRAM usage should be MUCH lower!")
        print(f"[BlockSwap] =====================================")

    # Attach tracking data to model_patcher for cleanup callback
    model_patcher.attachments['blockswap_tracking'] = blockswap_tracking
    if block_swap_debug:
        print(f"[BlockSwap] Block tracking attached for ON_CLEANUP callback")
        print(f"[BlockSwap] Tracking {len(blockswap_tracking.swapped_indices)} swapped blocks")
    
    # CRITICAL: Wrap unpatch_model to restore blocks BEFORE unpatching
    # This prevents "CUDA error: invalid argument" when moving partially-offloaded GGUF models
    _wrap_unpatch_model(model_patcher, block_swap_debug)


def _wrap_unpatch_model(model_patcher: Any, block_swap_debug: bool = False) -> None:
    """
    Wrap the model patcher's unpatch_model method to handle GGUF safely.
    
    When ComfyUI unloads a model (to free VRAM for another model), it calls
    unpatch_model which tries to move the ENTIRE model to CPU. But GGUF 
    tensors fail when you try to move them between devices.
    
    This wrapper:
    1. For GGUF models: Restores any swapped blocks to GPU, then lets ComfyUI-GGUF
       handle the unpatch with special GGUF-aware logic
    2. For native models: Restores swapped blocks to GPU before unpatch
    
    The key insight is that GGUF's GGMLTensor.to() method can fail with
    "CUDA error: invalid argument" when moving to the same device or when
    the tensor is in an inconsistent state.
    """
    # Check if already wrapped to prevent double-wrapping
    if hasattr(model_patcher, '_blockswap_unpatch_wrapped'):
        if block_swap_debug:
            print("[BlockSwap] unpatch_model already wrapped, skipping")
        return
    
    # Store the original method - keep a backup so we can always restore
    original_unpatch = model_patcher.unpatch_model
    model_patcher._blockswap_original_unpatch = original_unpatch
    
    def wrapped_unpatch_model(device_to=None, unpatch_weights=True):
        """Safe unpatch that handles GGUF and BlockSwap state."""
        tracking = model_patcher.attachments.get('blockswap_tracking')
        is_gguf = False
        
        if tracking is not None:
            is_gguf = getattr(tracking, 'is_gguf_model', False)
            blocks_restored = getattr(tracking, 'blocks_restored', False)
            successfully_swapped = getattr(tracking, 'successfully_swapped_indices', [])
            
            # Restore any swapped blocks to GPU before unpatch
            if not blocks_restored and len(successfully_swapped) > 0:
                if block_swap_debug:
                    print(f"[BlockSwap] PRE-UNPATCH: Restoring {len(successfully_swapped)} blocks to GPU")
                
                base_model = model_patcher.model
                unet = BlockManager.get_unet_from_model(base_model)
                
                if unet is not None and hasattr(unet, 'blocks'):
                    main_device = torch.device('cuda')
                    restored = 0
                    skipped = 0
                    
                    for block_idx in successfully_swapped:
                        try:
                            if block_idx < len(unet.blocks):
                                block = unet.blocks[block_idx]
                                
                                # Check if block is already on GPU to avoid GGUF errors
                                current_device = None
                                try:
                                    for param in block.parameters():
                                        current_device = param.device
                                        break
                                except (StopIteration, RuntimeError):
                                    pass
                                
                                if current_device is not None and current_device.type == 'cuda':
                                    # Already on GPU, skip the move
                                    skipped += 1
                                    continue
                                
                                block.to(main_device, non_blocking=False)
                                restored += 1
                        except Exception as e:
                            if block_swap_debug:
                                print(f"[BlockSwap] PRE-UNPATCH: Block {block_idx} restore failed: {str(e)[:50]}")
                    
                    if block_swap_debug:
                        if skipped > 0:
                            print(f"[BlockSwap] PRE-UNPATCH: Restored {restored}, skipped {skipped} (already on GPU)")
                        else:
                            print(f"[BlockSwap] PRE-UNPATCH: Restored {restored}/{len(successfully_swapped)} blocks")
                    
                    # Sync to ensure moves complete
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                
                tracking.blocks_restored = True
        
        # Clear tracking attachment to allow proper model release
        # This is critical for the "clear cache" button to work
        if 'blockswap_tracking' in model_patcher.attachments:
            del model_patcher.attachments['blockswap_tracking']
        
        # For GGUF models, we need special handling
        if is_gguf:
            if block_swap_debug:
                print("[BlockSwap] GGUF model detected - using safe unpatch")
            
            try:
                # Try calling the original unpatch
                result = original_unpatch(device_to=device_to, unpatch_weights=unpatch_weights)
                # Restore original unpatch method after successful call
                model_patcher.unpatch_model = original_unpatch
                return result
            except Exception as e:
                error_str = str(e)
                if "invalid argument" in error_str.lower() or "cuda" in error_str.lower():
                    # GGUF-specific error - try to recover
                    if block_swap_debug:
                        print(f"[BlockSwap] GGUF unpatch failed: {error_str[:80]}")
                        print("[BlockSwap] Attempting recovery - skipping problematic .to() calls")
                    
                    # Perform a minimal unpatch that skips the problematic model.to() call
                    try:
                        _safe_gguf_unpatch(model_patcher, device_to, unpatch_weights, block_swap_debug)
                        # Restore original unpatch method
                        model_patcher.unpatch_model = original_unpatch
                        return model_patcher.model
                    except Exception as recovery_error:
                        if block_swap_debug:
                            print(f"[BlockSwap] Recovery also failed: {str(recovery_error)[:80]}")
                        # Restore original unpatch method even on failure
                        model_patcher.unpatch_model = original_unpatch
                        # Re-raise the original error
                        raise e
                else:
                    # Non-GGUF error, restore and re-raise
                    model_patcher.unpatch_model = original_unpatch
                    raise
        else:
            # Native model - call original and restore
            result = original_unpatch(device_to=device_to, unpatch_weights=unpatch_weights)
            model_patcher.unpatch_model = original_unpatch
            return result
    
    # Replace the method and mark as wrapped
    model_patcher.unpatch_model = wrapped_unpatch_model
    model_patcher._blockswap_unpatch_wrapped = True
    
    if block_swap_debug:
        print("[BlockSwap] Wrapped unpatch_model for safe GGUF unloading")


def _safe_gguf_unpatch(model_patcher: Any, device_to: Any, unpatch_weights: bool, block_swap_debug: bool) -> None:
    """
    Perform a safe unpatch for GGUF models that skips problematic .to() calls.
    
    This is a minimal implementation that does the essential cleanup without
    the model.to(device_to) call that causes CUDA errors with GGUF tensors.
    """
    if block_swap_debug:
        print("[BlockSwap] Performing safe GGUF unpatch (skipping model.to())")
    
    # Call eject_model if available (unhooks the model)
    if hasattr(model_patcher, 'eject_model'):
        try:
            model_patcher.eject_model()
        except Exception as e:
            if block_swap_debug:
                print(f"[BlockSwap] eject_model failed (non-critical): {str(e)[:50]}")
    
    if unpatch_weights:
        # Unpatch hooks if available
        if hasattr(model_patcher, 'unpatch_hooks'):
            try:
                model_patcher.unpatch_hooks()
            except Exception as e:
                if block_swap_debug:
                    print(f"[BlockSwap] unpatch_hooks failed (non-critical): {str(e)[:50]}")
        
        # Unpin weights if available
        if hasattr(model_patcher, 'unpin_all_weights'):
            try:
                model_patcher.unpin_all_weights()
            except Exception as e:
                if block_swap_debug:
                    print(f"[BlockSwap] unpin_all_weights failed (non-critical): {str(e)[:50]}")
        
        # Handle lowvram state
        if hasattr(model_patcher.model, 'model_lowvram') and model_patcher.model.model_lowvram:
            # Skip the module iteration that does .to() - this is what causes the CUDA error
            model_patcher.model.model_lowvram = False
            model_patcher.model.lowvram_patch_counter = 0
        
        # Restore backup weights if any
        if hasattr(model_patcher, 'backup'):
            keys = list(model_patcher.backup.keys())
            for k in keys:
                try:
                    bk = model_patcher.backup[k]
                    if hasattr(bk, 'inplace_update') and bk.inplace_update:
                        import comfy.utils
                        comfy.utils.copy_to_param(model_patcher.model, k, bk.weight)
                    else:
                        import comfy.utils
                        comfy.utils.set_attr_param(model_patcher.model, k, bk.weight)
                except Exception as e:
                    if block_swap_debug:
                        print(f"[BlockSwap] backup restore failed for {k}: {str(e)[:30]}")
            model_patcher.backup.clear()
        
        # Clear object patches backup
        if hasattr(model_patcher, 'object_patches_backup'):
            for k in list(model_patcher.object_patches_backup.keys()):
                try:
                    import comfy.utils
                    comfy.utils.set_attr(model_patcher.model, k, model_patcher.object_patches_backup[k])
                except Exception:
                    pass
            model_patcher.object_patches_backup.clear()
    
    # Clear any blockswap attachments to allow proper model release
    if 'blockswap_tracking' in model_patcher.attachments:
        del model_patcher.attachments['blockswap_tracking']
    
    # Restore original unpatch method if we have it
    if hasattr(model_patcher, '_blockswap_original_unpatch'):
        model_patcher.unpatch_model = model_patcher._blockswap_original_unpatch
        delattr(model_patcher, '_blockswap_original_unpatch')
    
    # Remove the unpatch wrapper flag so model can be fully released
    if hasattr(model_patcher, '_blockswap_unpatch_wrapped'):
        delattr(model_patcher, '_blockswap_unpatch_wrapped')
    
    # Force GPU sync and cache clear
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    
    gc.collect()
    
    if block_swap_debug:
        print("[BlockSwap] Safe GGUF unpatch complete")


def cleanup_callback(model_patcher: Any) -> None:
    """
    ON_CLEANUP callback for comprehensive cleanup operations.

    Handles both GGUF and native models with appropriate cleanup strategies.
    Integrates with BlockSwapModelTracker for smart cleanup decisions.
    """
    try:
        tracking = model_patcher.attachments.get('blockswap_tracking')
        if tracking is None or tracking.cleanup_executed:
            return

        # Get tracker for smart cleanup decisions
        tracker = BlockSwapModelTracker.get_instance()
        model_id = id(model_patcher.model)

        # Get session_id from tracking if available
        session_id = getattr(tracking, 'session_id', None)

        # If no session, also check if tracker has this model
        if session_id is None:
            session_id = tracker.find_session_for_model(model_id)

        # Check with tracker for cleanup decision (pass model_patcher, not model_id)
        if session_id:
            decision = tracker.get_cleanup_decision(model_patcher)

            if decision == CleanupDecision.SKIP:
                if getattr(tracking, 'block_swap_debug', False):
                    print("[BlockSwap] Cleanup skipped (tracker decision)")
                return

            if decision == CleanupDecision.PRESERVE:
                if getattr(tracking, 'block_swap_debug', False):
                    print("[BlockSwap] Blocks preserved for next loop")
                # Clear tracking state but DON'T move/delete blocks
                _clear_callback_tracking(model_patcher)
                return

        tracking.cleanup_executed = True

        # Use getattr for safety in case attribute doesn't exist
        debug_enabled = getattr(tracking, 'block_swap_debug', False)

        if debug_enabled:
            print(f"[BlockSwap] ===== ON_CLEANUP CALLBACK EXECUTING =====")

        # Get model info
        base_model = model_patcher.model
        is_gguf = getattr(tracking, 'is_gguf_model', False)
        successfully_swapped = getattr(tracking, 'successfully_swapped_indices', [])

        if debug_enabled:
            print(f"[BlockSwap] Model type: {'GGUF' if is_gguf else 'Native'}")
            print(f"[BlockSwap] Successfully swapped blocks: {len(successfully_swapped)}")

        # Get UNet
        unet = BlockManager.get_unet_from_model(base_model)
        main_device = torch.device("cuda")

        # Phase 1: Synchronize GPU before any block operations
        if debug_enabled:
            print("[BlockSwap] Phase 1: Synchronizing GPU...")
        sync_gpu(debug_enabled)

        # Phase 2: Handle block cleanup based on model type
        if unet is not None and hasattr(unet, "blocks") and len(successfully_swapped) > 0:
            if is_gguf:
                # GGUF: Move blocks back to GPU (don't delete!)
                if debug_enabled:
                    print("[BlockSwap] Phase 2 (GGUF): Moving blocks back to GPU")

                moved_back = 0
                move_failures = 0

                for block_idx in successfully_swapped:
                    try:
                        if block_idx < len(unet.blocks):
                            # Move back to GPU with blocking (non_blocking=False) for safety
                            unet.blocks[block_idx].to(main_device, non_blocking=False)
                            moved_back += 1
                    except Exception as e:
                        move_failures += 1
                        if debug_enabled:
                            print(f"[BlockSwap] Block {block_idx} move-back failed: {str(e)[:80]}")

                if debug_enabled:
                    print(f"[BlockSwap] GGUF: Moved {moved_back}/{len(successfully_swapped)} blocks back to GPU")
                    if move_failures > 0:
                        print(f"[BlockSwap] GGUF: {move_failures} move-back failures (non-critical)")

                # Clear tracking but DON'T delete references for GGUF
                tracking.swapped_blocks_refs.clear()

            else:
                # NATIVE: Aggressive cleanup (delete references)
                if debug_enabled:
                    print("[BlockSwap] Phase 2 (Native): Deleting swapped block references")

                blocks_deleted = 0
                for block_idx in successfully_swapped:
                    try:
                        if block_idx in tracking.swapped_blocks_refs:
                            del tracking.swapped_blocks_refs[block_idx]
                            blocks_deleted += 1
                    except Exception as e:
                        if debug_enabled:
                            print(f"[BlockSwap] Failed to delete block {block_idx}: {str(e)[:80]}")

                if debug_enabled:
                    print(f"[BlockSwap] Native: Deleted {blocks_deleted}/{len(successfully_swapped)} block references")

                tracking.swapped_blocks_refs.clear()

        # Phase 3: Cleanup embeddings (same for both GGUF and Native)
        if debug_enabled:
            print("[BlockSwap] Phase 3: Cleaning up embedding references")

        for emb_name in list(tracking.embeddings_offloaded.keys()):
            try:
                del tracking.embeddings_offloaded[emb_name]
                if debug_enabled:
                    print(f"[BlockSwap] Deleted embedding: {emb_name}")
            except Exception as e:
                if debug_enabled:
                    print(f"[BlockSwap] Failed to delete embedding {emb_name}: {str(e)[:80]}")

        tracking.embeddings_offloaded.clear()

        # Phase 4: Clear tracking indices
        tracking.swapped_indices.clear()
        tracking.successfully_swapped_indices.clear()
        tracking.failed_to_swap_indices.clear()

        # Phase 5: Garbage collection
        if debug_enabled:
            print("[BlockSwap] Phase 5: Running garbage collection")
        gc.collect()

        # Phase 6: Clear device caches (AFTER operations complete)
        if debug_enabled:
            print("[BlockSwap] Phase 6: Clearing device caches")
        clear_device_caches(debug_enabled)

        if debug_enabled:
            print("[BlockSwap] ===== ON_CLEANUP COMPLETE =====")
            if is_gguf:
                print("[BlockSwap] GGUF: Blocks moved back to GPU safely (no deletion)")
            else:
                print("[BlockSwap] Native: All swapped block references freed from memory")

        # Mark cleanup done in tracker if session active
        if session_id:
            tracker.mark_cleanup_done(model_patcher)
        
        # CRITICAL: Clear the tracking attachment to allow model to be freed
        # This is what allows the "clear cache" button to work
        if 'blockswap_tracking' in model_patcher.attachments:
            del model_patcher.attachments['blockswap_tracking']
        
        # Restore original unpatch method if we wrapped it
        if hasattr(model_patcher, '_blockswap_original_unpatch'):
            model_patcher.unpatch_model = model_patcher._blockswap_original_unpatch
            delattr(model_patcher, '_blockswap_original_unpatch')
        
        # Remove unpatch wrapper flag
        if hasattr(model_patcher, '_blockswap_unpatch_wrapped'):
            delattr(model_patcher, '_blockswap_unpatch_wrapped')

    except Exception as e:
        print(f"[BlockSwap] CRITICAL ERROR in cleanup_callback: {str(e)}")
        import traceback
        traceback.print_exc()


def _clear_callback_tracking(model_patcher: Any) -> None:
    """Clear tracking state for preserve mode.

    This is used when the tracker decides to PRESERVE blocks for subsequent
    loops. We clear the tracking state but don't move or delete blocks.

    Args:
        model_patcher: The model patcher instance
    """
    tracking = model_patcher.attachments.get('blockswap_tracking')
    if tracking is None:
        return

    tracking.cleanup_executed = True
    tracking.swapped_indices.clear()
    tracking.successfully_swapped_indices.clear()
    tracking.failed_to_swap_indices.clear()
    tracking.embeddings_offloaded.clear()
