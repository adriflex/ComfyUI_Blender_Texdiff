import hashlib
import os

import comfy
import numpy as np
import torch
from PIL import Image


class ViewportColor:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image_path": ("STRING", {"image_upload": True})}}

    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "load_image"
    CATEGORY = "Blender TexDiff"

    def load_image(self, image_path):
        img = Image.open(image_path)

        if img.mode == "I":
            img = img.point(lambda i: i * (1 / 255))

        image = img.convert("RGB")
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image)[None,]

        if "A" in img.getbands():
            mask = np.array(img.getchannel("A")).astype(np.float32) / 255.0
            mask = 1.0 - torch.from_numpy(mask)
        else:
            mask = torch.zeros((64, 64), dtype=torch.float32, device="cpu")

        return (image, mask.unsqueeze(0))

    @classmethod
    def IS_CHANGED(s, image_path):
        if not os.path.exists(image_path):
            return ""
        m = hashlib.sha256()
        with open(image_path, "rb") as f:
            m.update(f.read())
        return m.digest().hex()


class ViewportDepth:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image_path": ("STRING", {"image_upload": True})}}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "load_image"
    CATEGORY = "Blender TexDiff"

    def load_image(self, image_path):
        img = Image.open(image_path)

        if img.mode == "I":
            img = img.point(lambda i: i * (1 / 255))

        image = img.convert("RGB")
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image)[None,]

        if "A" in img.getbands():
            mask = np.array(img.getchannel("A")).astype(np.float32) / 255.0
            mask = 1.0 - torch.from_numpy(mask)
        else:
            mask = torch.zeros((64, 64), dtype=torch.float32, device="cpu")

        return (image,)

    @classmethod
    def IS_CHANGED(s, image_path):
        if not os.path.exists(image_path):
            return ""
        m = hashlib.sha256()
        with open(image_path, "rb") as f:
            m.update(f.read())
        return m.digest().hex()


class Sampler_Setting:
    @classmethod
    def INPUT_TYPES(cls):
        # input = float, string and int
        return {
            "required": {
                "seed": ("INT", {"min": 0}),
                "sampler_steps": ("INT", {"min": 0}),
                "cgf_scale": ("FLOAT", {"min": 0.0}),
                # "sampler_name": ("STRING", {"default": ""}),
                # "scheduler": ("STRING", {"default": ""}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "denoise": ("FLOAT", {"min": 0.0, "max": 1.0}),
            }
        }

    RETURN_TYPES = (
        "INT",
        "INT",
        "FLOAT",
        comfy.samplers.KSampler.SAMPLERS,
        comfy.samplers.KSampler.SCHEDULERS,
        "FLOAT",
    )
    FUNCTION = "get_sampler_setting"
    CATEGORY = "Blender TexDiff"

    def get_sampler_setting(
        self, sampler_steps, cgf_scale, sampler_name, scheduler, denoise
    ):
        return (sampler_steps, cgf_scale, sampler_name, scheduler, denoise)


NODE_CLASS_MAPPINGS = {
    "ViewportColor": ViewportColor,
    "ViewportDepth": ViewportDepth,
    "Sampler_Setting": Sampler_Setting,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ViewportColor": "From blender viewport color",
    "ViewportDepth": "From blender Viewport depth",
    "Sampler_Setting": "From blender sampler setting",
}
