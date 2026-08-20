try:
    from typing import override
except ImportError:
    from typing_extensions import override

from comfy_api.latest import ComfyExtension, io

from .cosmos_temporal_reference import ApplyCosmosReferenceLatent, AnimaEditModelLoader


class CosmosReferenceExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [ApplyCosmosReferenceLatent, AnimaEditModelLoader]


async def comfy_entrypoint():
    return CosmosReferenceExtension()
