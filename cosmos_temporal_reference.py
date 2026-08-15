from typing import Any, Callable

import torch

import comfy.conds
import comfy.patcher_extension
from comfy.model_base import Anima
from comfy.model_patcher import ModelPatcher
from comfy_api.latest import io

MAX_REF_LATENTS = 2
COND_REF_LATENTS_KEY = "ref_latents"
TEMPORAL_REFERENCE_WRAPPER_KEY = "cosmos_temporal_reference"


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
                io.Autogrow.Input(
                    "ref_latents",
                    template=io.Autogrow.TemplateNames(
                        io.Latent.Input("ref_latent"),
                        names=[f"ref_latent_{i}" for i in range(1, MAX_REF_LATENTS + 1)],
                        min=0,
                    ),
                    tooltip=f"Reference latent(s) for generation. Up to {MAX_REF_LATENTS} latents.",
                ),
            ],
            outputs=[
                io.Model.Output(),
            ],
        )

    @classmethod
    def execute(cls, **kwargs) -> io.NodeOutput:
        model: ModelPatcher = kwargs["model"]
        ref_latents: dict[str, dict[str, Any]] = kwargs["ref_latents"]
        if 'latent' in kwargs:
            ref_latents['latent_for_compatibility'] = kwargs["latent"]
        m = model.clone()
        model_type = type(m.model)

        if issubclass(model_type, Anima):
            extra_conds = m.get_model_object("extra_conds")
            process_latent_in = m.get_model_object("process_latent_in")
            m.add_object_patch(
                "extra_conds",
                cosmos_extra_conds_reference(extra_conds, process_latent_in, ref_latents),
            )
            m.add_wrapper_with_key(
                comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL,
                TEMPORAL_REFERENCE_WRAPPER_KEY,
                cosmos_diffusion_reference_wrapper,
            )

        return io.NodeOutput(m)


def cosmos_extra_conds_reference(
    extra_conds: Callable[..., dict],
    process_latent_in: Callable[..., torch.Tensor],
    ref_latents: dict[str, dict[str, Any]] | None = None,
):
    def _anima_extra_conds_reference(**kwargs):
        out = extra_conds(**kwargs)
        if ref_latents is not None:
            latents = [process_latent_in(l["samples"]) for l in ref_latents.values()]
            out[COND_REF_LATENTS_KEY] = comfy.conds.CONDList(latents)

        return out

    return _anima_extra_conds_reference


def cosmos_diffusion_reference_wrapper(executor, *args, **kwargs):
    x: torch.Tensor = args[0]
    x_temporal_dim = x.shape[2]
    ref_latents = kwargs.get(COND_REF_LATENTS_KEY)

    newargs = list(args)

    if ref_latents is not None:
        refs = list(ref_latents)

        # Reference 1
        if len(refs) >= 1:
            ref_latent_1 = refs[0]
            if ref_latent_1.ndim == 4:
                ref_latent_1 = ref_latent_1.unsqueeze(2)
            ref1 = ref_latent_1.to(dtype=x.dtype, device=x.device)
            x = torch.cat([x, ref1], dim=2)

        # Reference 2
        if len(refs) >= 2:
            ref_latent_2 = refs[1]
            if ref_latent_2.ndim == 4:
                ref_latent_2 = ref_latent_2.unsqueeze(2)
            ref2 = ref_latent_2.to(dtype=x.dtype, device=x.device)
            x = torch.cat([x, ref2], dim=2)

    newargs[0] = x

    result = executor(*newargs, **kwargs)

    # Remove both reference temporal sections and return only the original x.
    result = result[:, :, :x_temporal_dim]

    return result
