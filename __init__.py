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
        
        # 1. 拦截 extra_conds 以便从 Conditioning 中提取参考图
        original_extra_conds = m.model.extra_conds

        def custom_extra_conds(self, **kwargs):
            out = original_extra_conds(**kwargs)
            
            ref_latents = kwargs.get("reference_latents", None)
            if ref_latents is not None:
                latents = []
                # 遍历传入的参考图，并调用模型内置方法应用 VAE Scale
                for lat in ref_latents:
                    latents.append(self.process_latent_in(lat))
                # 打包成 ComfyUI 的规范格式输出给下一步
                out['ref_latents'] = comfy.conds.CONDList(latents)
            return out

        # 将新方法绑定到模型实例上，跑完自动恢复
        bound_extra_conds = types.MethodType(custom_extra_conds, m.model)
        m.add_object_patch("extra_conds", bound_extra_conds)

        # 2. 应用 UNet 包装器
        prev_wrapper = m.model_options.get("model_function_wrapper", None)

        def ref_latent_unet_wrapper(model_apply, model_kwargs):
            c_dict = model_kwargs.get("c", {})
            
            # --- 收集参考图 ---
            # 来源 A: 来自 extra_conds (Conditioning 注入)
            ref_from_cond = c_dict.get("ref_latents", [])
            if not isinstance(ref_from_cond, (list, comfy.conds.CONDList)):
                ref_from_cond = [ref_from_cond]
            
            # 来源 B: 直接从模型对象中获取 (通过 Object Patch 注入)
            model_obj = getattr(model_apply, "__self__", None)
            ref_from_options = getattr(model_obj, "cosmos_ref_latents", [])
            
            all_refs = list(ref_from_cond) + list(ref_from_options)
            
            if len(all_refs) > 0:
                orig_x = model_kwargs.get("input")
                new_x = orig_x
                
                # 记录原始的时间维度 T
                orig_T = orig_x.shape[2] 
                
                for ref in all_refs:
                    if isinstance(ref, torch.Tensor):
                        # Cosmos / Flux2 风格升维
                        if ref.ndim == 4:
                            ref = ref.unsqueeze(2)
                        
                        ref = ref.to(dtype=new_x.dtype, device=new_x.device)
                        
                        # 匹配 CFG 批处理维度
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

                # 执行原模型链条
                if prev_wrapper is not None:
                    out = prev_wrapper(model_apply, model_kwargs)
                else:
                    mk = model_kwargs.copy()
                    x_val = mk.pop("input")
                    t_val = mk.pop("timestep")
                    cond_dict = mk.pop("c", {}) 
                    out = model_apply(x_val, t_val, **cond_dict, **mk)
                    
                # 裁剪回原始长度
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

class ApplyCosmosReferenceLatent:
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
        # 1. 确保基础补丁已应用
        patcher = ApplyCosmosReferenceModelPatch()
        (m,) = patcher.patch(model)
        
        # 2. 注入当前 Latent
        samples = latent["samples"]
        
        # 处理 Latent 缩放
        if hasattr(m.model, "process_latent_in"):
            samples = m.model.process_latent_in(samples)
        
        # 3. 维护参考图列表
        # 使用 model_options 存储列表以确保节点链条间的隔离，避免读取到模型对象的全局缓存
        ref_list = list(m.model_options.get("cosmos_ref_latents_list", []))
        ref_list.append(samples)
        
        # 更新 model_options 以便链式调用
        m.model_options["cosmos_ref_latents_list"] = ref_list
        
        # 将最终列表通过 Object Patch 同步给模型对象，供采样时的 Wrapper 读取
        # 注意：这里的 key 必须与 Wrapper 中 getattr 的 key (cosmos_ref_latents) 一致
        m.add_object_patch("cosmos_ref_latents", ref_list)
        
        return (m,)

NODE_CLASS_MAPPINGS = {
    "ApplyCosmosReferenceModelPatch": ApplyCosmosReferenceModelPatch,
    "ApplyCosmosReferenceLatent": ApplyCosmosReferenceLatent
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ApplyCosmosReferenceModelPatch": "Apply Cosmos Reference Model Patch",
    "ApplyCosmosReferenceLatent": "Apply Cosmos Reference Latent"
}
