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


class ViewportDepth:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image_path": ("STRING", {"image_upload": True})}}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "load_image"
    CATEGORY = "Blender TexDiff"

    def load_image(self, image_path):
        img = Image.open(image_path)

        image = np.array(img).astype(np.float32) / 255.0
        image = torch.from_numpy(image)[None,]

        return (image,)


NODE_CLASS_MAPPINGS = {
    "ViewportColor": ViewportColor,
    "ViewportDepth": ViewportDepth,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ViewportColor": "Blender viewport color",
    "ViewportDepth": "Blender Viewport depth",
}
