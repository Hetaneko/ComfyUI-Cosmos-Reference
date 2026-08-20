from typing import Any, Callable

import torch

import comfy.conds
import comfy.patcher_extension
from comfy.model_base import Anima
from comfy.model_patcher import ModelPatcher
from comfy_api.latest import io

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
                io.Latent.Input("layout_latent", optional=True, tooltip="Optional layout reference latent."),
                io.Latent.Input("character_latent", optional=True, tooltip="Optional character reference latent."),
                io.Latent.Input("background_latent", optional=True, tooltip="Optional background reference latent."),
            ],
            outputs=[
                io.Model.Output(),
            ],
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
            m.add_object_patch(
                "extra_conds",
                cosmos_extra_conds_reference(extra_conds, process_latent_in, ref_latents),
            )
            m.add_wrapper_with_key(
                comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL,
                TEMPORAL_REFERENCE_WRAPPER_KEY,
                cosmos_diffusion_reference_wrapper(tuple(ref_latents.keys())),
            )

        return io.NodeOutput(m)


def cosmos_extra_conds_reference(
    extra_conds: Callable[..., dict],
    process_latent_in: Callable[..., torch.Tensor],
    ref_latents: dict[str, dict[str, Any]] | None = None,
):
    def _anima_extra_conds_reference(**kwargs):
        out = extra_conds(**kwargs)
        if ref_latents:
            latents = [process_latent_in(l["samples"]) for l in ref_latents.values()]
            out[COND_REF_LATENTS_KEY] = comfy.conds.CONDList(latents)

        return out

    return _anima_extra_conds_reference


CONTROL_REF_IDS = {
    "layout_latent": 1,
    "character_latent": 2,
    "background_latent": 3,
}


def cosmos_diffusion_reference_wrapper(active_ref_names: tuple[str, ...]):
    def _cosmos_diffusion_reference_wrapper(executor, *args, **kwargs):
        x: torch.Tensor = args[0]
        x_temporal_dim = x.shape[2]
        ref_latents = kwargs.get(COND_REF_LATENTS_KEY)

        newargs = list(args)

        if ref_latents is not None:
            refs = list(ref_latents)
            active_control_ref_ids = torch.tensor(
                [CONTROL_REF_IDS[name] for name in active_ref_names],
                dtype=torch.long,
                device=x.device,
            )

            for ref_latent in refs:
                if ref_latent.ndim == 4:
                    ref_latent = ref_latent.unsqueeze(2)
                ref = ref_latent.to(dtype=x.dtype, device=x.device)
                x = torch.cat([x, ref], dim=2)

            kwargs["active_control_ref_ids"] = active_control_ref_ids

        newargs[0] = x

        result = executor(*newargs, **kwargs)

        # Remove reference temporal sections and return only the original x.
        result = result[:, :, :x_temporal_dim]

        return result

    return _cosmos_diffusion_reference_wrapper
