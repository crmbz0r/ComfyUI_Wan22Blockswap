/**
 * WAN22Blockswap UI Extension
 * 
 * Dynamically disables dtype and FP8 widgets when GGUF model type is selected,
 * since those options have no effect on GGUF quantized models (they have their
 * own quantization format).
 * 
 * Implementation approach:
 * - Wraps the widget's mouse handler to block interaction when disabled
 * - Wraps the widget's draw method to render with reduced opacity when disabled
 * - Sets a custom 'disabled' property that the wrappers check
 */

import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "wan22blockswap.conditional_widgets",
    
    async nodeCreated(node) {
        // Target both loader nodes
        const targetNodes = ["WAN22BlockSwapLoader", "WANModelLoader"];
        
        if (!targetNodes.includes(node.comfyClass)) {
            return;
        }
        
        // Find the model_type widget
        const modelTypeWidget = node.widgets?.find(w => w.name === "model_type");
        if (!modelTypeWidget) {
            return;
        }
        
        // Widgets to disable when GGUF is selected
        const affectedWidgetNames = ["fp8_optimization", "weight_dtype"];
        
        /**
         * Wrap a widget to support disabled state with visual feedback.
         * This patches the widget's mouse and draw methods to:
         * - Block mouse interaction when disabled
         * - Draw with reduced opacity when disabled
         */
        const wrapWidgetForDisable = (widget) => {
            if (widget._disableWrapped) return; // Already wrapped
            widget._disableWrapped = true;
            widget.disabled = false;
            
            // Store original mouse handler
            const origMouse = widget.mouse;
            widget.mouse = function(event, pos, node) {
                // Block all mouse interaction when disabled
                if (this.disabled) {
                    return false;
                }
                if (origMouse) {
                    return origMouse.call(this, event, pos, node);
                }
                return false;
            };
            
            // Store original draw method (may be undefined for combo widgets)
            const origDraw = widget.draw?.bind(widget);
            
            // Only wrap if there's an original draw method
            // Combo widgets use default LiteGraph drawing, we'll overlay on that
            if (origDraw) {
                widget.draw = function(ctx, node, width, y, height) {
                    if (this.disabled) {
                        // Save context state and apply transparency
                        ctx.save();
                        ctx.globalAlpha = 0.4;
                    }
                    
                    // Call original draw
                    origDraw(ctx, node, width, y, height);
                    
                    if (this.disabled) {
                        // Restore context state
                        ctx.restore();
                        
                        // Draw a subtle overlay with "N/A" indicator
                        ctx.save();
                        ctx.fillStyle = "rgba(80, 80, 80, 0.3)";
                        ctx.fillRect(0, y, width, height);
                        
                        // Add small "GGUF" indicator on right side
                        ctx.fillStyle = "rgba(255, 180, 100, 0.8)";
                        ctx.font = "9px Arial";
                        ctx.textAlign = "right";
                        ctx.fillText("(GGUF)", width - 25, y + height / 2 + 3);
                        ctx.restore();
                    }
                };
            } else {
                // For widgets without custom draw, add a post-draw overlay using onDrawForeground
                // We'll use a different approach: store disabled state and check during node draw
                widget._noCustomDraw = true;
            }
            
            // Wrap callback to prevent value changes when disabled
            const origCallback = widget.callback;
            widget.callback = function(value) {
                if (this.disabled) {
                    return; // Block callback when disabled
                }
                if (origCallback) {
                    origCallback.call(this, value);
                }
            };
        };
        
        /**
         * Update widget states based on model_type selection.
         * When GGUF is selected, dtype and FP8 options are disabled since
         * GGUF models use their own quantization format.
         */
        const updateWidgetStates = (modelType) => {
            const isGGUF = modelType === "gguf";
            
            for (const widgetName of affectedWidgetNames) {
                const widget = node.widgets?.find(w => w.name === widgetName);
                if (widget) {
                    // Ensure widget is wrapped for disable support
                    wrapWidgetForDisable(widget);
                    
                    // Set disabled state
                    widget.disabled = isGGUF;
                }
            }
            
            // Force redraw to show updated state
            node.setDirtyCanvas(true, true);
        };
        
        // Store original callback for model_type widget
        const origCallback = modelTypeWidget.callback;
        
        // Override callback to update widget states on change
        modelTypeWidget.callback = function(value) {
            // Call original callback if exists
            if (origCallback) {
                origCallback.call(this, value);
            }
            
            updateWidgetStates(value);
        };
        
        // Initialize state on node creation
        // Use setTimeout to ensure widgets are fully initialized
        setTimeout(() => {
            updateWidgetStates(modelTypeWidget.value);
        }, 0);
    },
    
    async loadedGraphNode(node) {
        // Target both loader nodes
        const targetNodes = ["WAN22BlockSwapLoader", "WANModelLoader"];
        
        if (!targetNodes.includes(node.comfyClass)) {
            return;
        }
        
        // Find the model_type widget
        const modelTypeWidget = node.widgets?.find(w => w.name === "model_type");
        if (!modelTypeWidget) {
            return;
        }
        
        const affectedWidgetNames = ["fp8_optimization", "weight_dtype"];
        
        /**
         * Wrap a widget to support disabled state (same as in nodeCreated)
         */
        const wrapWidgetForDisable = (widget) => {
            if (widget._disableWrapped) return;
            widget._disableWrapped = true;
            widget.disabled = false;
            
            const origMouse = widget.mouse;
            widget.mouse = function(event, pos, node) {
                if (this.disabled) return false;
                if (origMouse) return origMouse.call(this, event, pos, node);
                return false;
            };
            
            const origDraw = widget.draw?.bind(widget);
            if (origDraw) {
                widget.draw = function(ctx, node, width, y, height) {
                    if (this.disabled) {
                        ctx.save();
                        ctx.globalAlpha = 0.4;
                    }
                    
                    origDraw(ctx, node, width, y, height);
                    
                    if (this.disabled) {
                        ctx.restore();
                        ctx.save();
                        ctx.fillStyle = "rgba(80, 80, 80, 0.3)";
                        ctx.fillRect(0, y, width, height);
                        ctx.fillStyle = "rgba(255, 180, 100, 0.8)";
                        ctx.font = "9px Arial";
                        ctx.textAlign = "right";
                        ctx.fillText("(GGUF)", width - 25, y + height / 2 + 3);
                        ctx.restore();
                    }
                };
            } else {
                widget._noCustomDraw = true;
            }
            
            const origCallback = widget.callback;
            widget.callback = function(value) {
                if (this.disabled) return;
                if (origCallback) origCallback.call(this, value);
            };
        };
        
        // Update states for loaded graph
        const isGGUF = modelTypeWidget.value === "gguf";
        
        for (const widgetName of affectedWidgetNames) {
            const widget = node.widgets?.find(w => w.name === widgetName);
            if (widget) {
                wrapWidgetForDisable(widget);
                widget.disabled = isGGUF;
            }
        }
        
        // Ensure model_type callback also updates states
        const origCallback = modelTypeWidget.callback;
        modelTypeWidget.callback = function(value) {
            if (origCallback) {
                origCallback.call(this, value);
            }
            
            const isGGUF = value === "gguf";
            for (const widgetName of affectedWidgetNames) {
                const widget = node.widgets?.find(w => w.name === widgetName);
                if (widget) {
                    wrapWidgetForDisable(widget);
                    widget.disabled = isGGUF;
                }
            }
            node.setDirtyCanvas(true, true);
        };
        
        // Force initial redraw
        node.setDirtyCanvas(true, true);
    }
});
