from typing import Any, Callable
import torch
import safetensors.torch
import folder_paths
import comfy.sd
import comfy.conds
import comfy.patcher_extension
from comfy.model_base import Anima
from comfy.model_patcher import ModelPatcher
from comfy_api.latest import io

COND_REF_LATENTS_KEY = "ref_latents"
TEMPORAL_REFERENCE_WRAPPER_KEY = "cosmos_temporal_reference"

CONTROL_REF_IDS = {
    "layout_latent": 1,
    "character_latent": 2,
    "background_latent": 3,
}

# -------------------------------------------------------------------------
# 1. THE CUSTOM LOADER 
# -------------------------------------------------------------------------
class AnimaEditModelLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="AnimaEditModelLoader",
            display_name="Load Anima Edit Model",
            category="model/loaders",
            inputs=[
                io.Combo.Input("unet_name", options=folder_paths.get_filename_list("diffusion_models")),
                io.Combo.Input("weight_dtype", options=["default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2"], advanced=True)
            ],
            outputs=[io.Model.Output("model")],
        )

    @classmethod
    def execute(cls, **kwargs) -> io.NodeOutput:
        unet_name = kwargs["unet_name"]
        weight_dtype = kwargs["weight_dtype"]

        model_options = {}
        if weight_dtype == "fp8_e4m3fn":
            model_options["dtype"] = torch.float8_e4m3fn
        elif weight_dtype == "fp8_e4m3fn_fast":
            model_options["dtype"] = torch.float8_e4m3fn
            model_options["fp8_optimizations"] = True
        elif weight_dtype == "fp8_e5m2":
            model_options["dtype"] = torch.float8_e5m2

        unet_path = folder_paths.get_full_path_or_raise("diffusion_models", unet_name)
        model = comfy.sd.load_diffusion_model(unet_path, model_options=model_options)

        sd = safetensors.torch.load_file(unet_path, device="cpu")
        
        layout_tag, character_tag, background_tag = None, None, None
        for key, tensor in sd.items():
            if "layout_tag" in key: layout_tag = tensor
            if "character_tag" in key: character_tag = tensor
            if "background_tag" in key: background_tag = tensor

        # Attach directly to the underlying PyTorch model to avoid waiting-room errors
        if layout_tag is not None:
            model.model.layout_tag = layout_tag.to(model.load_device)
            model.model.character_tag = character_tag.to(model.load_device)
            model.model.background_tag = background_tag.to(model.load_device)

        return io.NodeOutput(model)


# -------------------------------------------------------------------------
# 2. THE REFERENCE NODE (Latents Only)
# -------------------------------------------------------------------------
class ApplyCosmosReferenceLatent(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="ApplyCosmosReferenceLatent",
            search_aliases=["cosmos reference", "anima reference"],
            display_name="Apply Cosmos Reference Latent",
            category="conditioning",
            inputs=[
                io.Model.Input("model"),
                io.Latent.Input("layout_latent", optional=True),
                io.Latent.Input("character_latent", optional=True),
                io.Latent.Input("background_latent", optional=True),
            ],
            outputs=[io.Model.Output()],
        )

    @classmethod
    def execute(cls, **kwargs) -> io.NodeOutput:
        model: ModelPatcher = kwargs["model"]
        
        ref_latents = {
            name: kwargs[name]
            for name in ("layout_latent", "character_latent", "background_latent")
            if kwargs.get(name) is not None
        }

        m = model.clone()
        model_type = type(m.model)

        if issubclass(model_type, Anima):
            extra_conds = m.get_model_object("extra_conds")
            process_latent_in = m.get_model_object("process_latent_in")
            
            m.add_object_patch("extra_conds", cosmos_extra_conds_reference(extra_conds, process_latent_in, ref_latents))
            
            # --- DEBUG: CHECK IF TAGS ARE ACTUALLY LOADED AND NOT ZERO ---
            tags = {
                1: getattr(m.model, "layout_tag", None),
                2: getattr(m.model, "character_tag", None),
                3: getattr(m.model, "background_tag", None),
            }
            print("\n" + "="*50)
            print("DEBUG STAGE 1: CHECKING LOADED TAGS")
            for id, name in [(1, "Layout"), (2, "Character"), (3, "Background")]:
                if tags[id] is None:
                    print(f"-> {name} Tag (ID {id}): MISSING (None)")
                else:
                    tag_sum = tags[id].sum().item()
                    print(f"-> {name} Tag (ID {id}): FOUND | Shape: {tags[id].shape} | Sum of weights: {tag_sum:.5f}")
            print("="*50 + "\n")
            
            raw_model = getattr(m.model, "diffusion_model", m.model)
            
            m.add_wrapper_with_key(
                comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL,
                TEMPORAL_REFERENCE_WRAPPER_KEY,
                cosmos_diffusion_reference_wrapper(tuple(ref_latents.keys()), tags, raw_model),
            )

        return io.NodeOutput(m)


def cosmos_extra_conds_reference(extra_conds, process_latent_in, ref_latents):
    def _anima_extra_conds_reference(**kwargs):
        out = extra_conds(**kwargs)
        if ref_latents:
            latents = [process_latent_in(l["samples"]) for l in ref_latents.values()]
            out[COND_REF_LATENTS_KEY] = comfy.conds.CONDList(latents)
        return out
    return _anima_extra_conds_reference


def cosmos_diffusion_reference_wrapper(active_ref_names, tags, raw_model):
    def _cosmos_diffusion_reference_wrapper(executor, *args, **kwargs):
        x: torch.Tensor = args[0]
        x_temporal_dim = x.shape[2]
        ref_latents = kwargs.get(COND_REF_LATENTS_KEY)

        newargs = list(args)

        if ref_latents is not None:
            refs = list(ref_latents)
            active_ids = [CONTROL_REF_IDS[name] for name in active_ref_names]

            for ref_latent in refs:
                if ref_latent.ndim == 4:
                    ref_latent = ref_latent.unsqueeze(2)
                ref = ref_latent.to(dtype=x.dtype, device=x.device)
                
                if ref.shape[0] != x.shape[0]:
                    ref = ref.repeat(x.shape[0], 1, 1, 1, 1)
                
                x = torch.cat([x, ref], dim=2)

            newargs[0] = x

            hook_handle = [None]
            
            def pre_hook(module, block_args):
                tokens = block_args[0].clone() 
                target_t = x_temporal_dim 
                
                # --- DEBUG: INSIDE THE HOOK ---
                print("\n" + "="*50)
                print("DEBUG STAGE 2: INSIDE THE PARASITE HOOK")
                print(f"-> Tokens Shape: {tokens.shape}")
                print(f"-> Target Temporal Index: {target_t}")
                print(f"-> Active IDs to inject: {active_ids}")
                
                for idx, ref_id in enumerate(active_ids, start=target_t):
                    tag_weight = tags.get(ref_id)
                    if tag_weight is not None:
                        print(f"-> Injecting Tag ID {ref_id} at token index {idx}")
                        tag = tag_weight.to(device=tokens.device, dtype=tokens.dtype)
                        
                        try:
                            # Try to add it, if shapes misalign, it will crash and tell us!
                            tokens[:, idx] = tokens[:, idx] + tag.view(1, 1, 1, -1)
                            print(f"-> SUCCESS: Tag ID {ref_id} injected!")
                        except Exception as e:
                            print(f"-> CRITICAL SHAPE ERROR on injection: {e}")
                    else:
                        print(f"-> FAILED: Tag ID {ref_id} is None!")
                print("="*50 + "\n")
                
                if hook_handle[0] is not None:
                    hook_handle[0].remove()
                    
                new_block_args = list(block_args)
                new_block_args[0] = tokens
                return tuple(new_block_args)

            try:
                if hasattr(raw_model, "blocks"):
                    first_block = raw_model.blocks[0]
                elif hasattr(raw_model, "transformer") and hasattr(raw_model.transformer, "blocks"):
                    first_block = raw_model.transformer.blocks[0]
                else:
                    first_block = None
                    
                if first_block is not None:
                    hook_handle[0] = first_block.register_forward_pre_hook(pre_hook)
            except Exception as e:
                pass 

        result = executor(*newargs, **kwargs)
        result = result[:, :, :x_temporal_dim]
        return result

    return _cosmos_diffusion_reference_wrapper
