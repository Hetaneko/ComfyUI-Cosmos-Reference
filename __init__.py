import torch
import comfy.conds


class ApplyCosmosReferenceModelPatch:
    """Apply the reference-latent UNet wrapper once (idempotent).
    Usually not needed standalone — ApplyCosmosReferenceLatent handles this automatically.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"model": ("MODEL",)}}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "Cosmos/Reference"

    def patch(self, model):
        m = model.clone()

        # Already patched → no-op (avoid nested wrappers)
        if m.model_options.get("cosmos_ref_patched", False):
            return (m,)

        # Shared mutable list — all subsequent ApplyCosmosReferenceLatent nodes
        # will append to this same list via model_options.
        refs: list = []
        m.model_options["cosmos_ref_accum"] = refs
        m.model_options["cosmos_ref_patched"] = True

        def _wrapper(model_apply, model_kwargs):
            x = model_kwargs.pop("input")
            t = model_kwargs.pop("timestep")
            c = model_kwargs.pop("c", {})

            if not refs:
                return model_apply(x, t, **c, **model_kwargs)

            orig_T = x.shape[2]
            new_x = x

            for ref in refs:
                if ref.ndim == 4:
                    ref = ref.unsqueeze(2)
                ref = ref.to(dtype=new_x.dtype, device=new_x.device)

                bs_x, bs_ref = new_x.shape[0], ref.shape[0]
                if bs_ref != bs_x:
                    if bs_x % bs_ref == 0:
                        ref = ref.repeat(bs_x // bs_ref, 1, 1, 1, 1)
                    else:
                        ref = ref.expand(bs_x, -1, -1, -1, -1)
                new_x = torch.cat([new_x, ref], dim=2)

            out = model_apply(new_x, t, **c, **model_kwargs)
            return out[:, :, :orig_T, :, :]

        m.set_model_unet_function_wrapper(_wrapper)
        return (m,)


class ApplyCosmosReferenceLatent:
    """Add a reference latent.  Can be chained: each call clones the model
    and appends the latent without creating nested wrappers."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "latent": ("LATENT",),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "Cosmos/Reference"

    def apply(self, model, latent):
        m = model.clone()

        # Ensure the wrapper is installed (idempotent)
        if not m.model_options.get("cosmos_ref_patched", False):
            (m,) = ApplyCosmosReferenceModelPatch().patch(m)

        samples = latent["samples"]
        if hasattr(m.model, "process_latent_in"):
            samples = m.model.process_latent_in(samples)

        m.model_options["cosmos_ref_accum"].append(samples)
        return (m,)


NODE_CLASS_MAPPINGS = {
    "ApplyCosmosReferenceModelPatch": ApplyCosmosReferenceModelPatch,
    "ApplyCosmosReferenceLatent": ApplyCosmosReferenceLatent,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ApplyCosmosReferenceModelPatch": "Apply Cosmos Reference Model Patch",
    "ApplyCosmosReferenceLatent": "Apply Cosmos Reference Latent",
}
