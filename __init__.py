import torch
import types
import comfy.conds

class ApplyCosmosReferenceModelPatch:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "Cosmos/Reference"

    def patch(self, model):
        m = model.clone() 
        
        original_extra_conds = m.model.extra_conds

        def custom_extra_conds(self, **kwargs):
            out = original_extra_conds(**kwargs)
            
            ref_latents = kwargs.get("reference_latents", None)
            if ref_latents is not None:
                latents =[]
                # 遍历传入的参考图，并调用模型内置方法应用 VAE Scale
                for lat in ref_latents:
                    latents.append(self.process_latent_in(lat))
                # 打包成 ComfyUI 的规范格式输出给下一步
                out['ref_latents'] = comfy.conds.CONDList(latents)
            return out

        # 将新方法绑定到模型实例上，跑完自动恢复
        bound_extra_conds = types.MethodType(custom_extra_conds, m.model)
        m.add_object_patch("extra_conds", bound_extra_conds)

        prev_wrapper = m.model_options.get("model_function_wrapper", None)

        def ref_latent_unet_wrapper(model_apply, model_kwargs):
            c_dict = model_kwargs.get("c", {})
            
            # 从 extra_conds 传递过来的缩放后的 latent
            ref_latents = c_dict.get("ref_latents", None)
            
            if ref_latents is not None:
                orig_x = model_kwargs.get("input")
                new_x = orig_x
                
                # 记录原始的时间维度 T，非常重要！
                orig_T = orig_x.shape[2] 
                
                refs = ref_latents if isinstance(ref_latents, list) else [ref_latents]
                    
                for ref in refs:
                    if isinstance(ref, torch.Tensor):
                        # Cosmos / Flux2 风格升维
                        if ref.ndim == 4:
                            ref = ref.unsqueeze(2)
                        
                        ref = ref.to(dtype=new_x.dtype, device=new_x.device)
                        
                        # 匹配 CFG 批处理维度，防止 batch size 不匹配崩溃
                        bs_x = new_x.shape[0]
                        bs_ref = ref.shape[0]
                        if bs_ref != bs_x:
                            if bs_x % bs_ref == 0:
                                ref = ref.repeat(bs_x // bs_ref, 1, 1, 1, 1)
                            else:
                                ref = ref.expand(bs_x, -1, -1, -1, -1)
                        
                        # 沿时间轴拼接参考图片
                        new_x = torch.cat([new_x, ref], dim=2)
                
                model_kwargs['input'] = new_x

                # -------------------------
                # 执行原模型链条
                # -------------------------
                if prev_wrapper is not None:
                    out = prev_wrapper(model_apply, model_kwargs)
                else:
                    mk = model_kwargs.copy()
                    x_val = mk.pop("input")
                    t_val = mk.pop("timestep")
                    cond_dict = mk.pop("c", {}) 
                    out = model_apply(x_val, t_val, **cond_dict, **mk)
                    
                out = out[:, :, :orig_T, :, :]
                return out
                
            else:
                # 如果没有接参考图，走正常流程
                if prev_wrapper is not None:
                    return prev_wrapper(model_apply, model_kwargs)
                else:
                    mk = model_kwargs.copy()
                    x_val = mk.pop("input")
                    t_val = mk.pop("timestep")
                    cond_dict = mk.pop("c", {}) 
                    return model_apply(x_val, t_val, **cond_dict, **mk)

        m.set_model_unet_function_wrapper(ref_latent_unet_wrapper)

        return (m,)



NODE_CLASS_MAPPINGS = {
    "ApplyCosmosReferenceModelPatch": ApplyCosmosReferenceModelPatch
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ApplyCosmosReferenceModelPatch": "Apply Cosmos Reference Model Patch"
}