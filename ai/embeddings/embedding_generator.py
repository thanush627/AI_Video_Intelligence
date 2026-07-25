from pathlib import Path

import numpy as np
import torch
from PIL import Image

from ai.embeddings.clip_loader import CLIPLoader


class EmbeddingGenerator:

    def __init__(self):

        self.clip = CLIPLoader()

        self.model = self.clip.get_model()

        self.preprocess = self.clip.get_preprocess()

        self.device = self.clip.get_device()

    def generate_image_embedding(self, image_path):

        image_path = Path(image_path)

        if not image_path.exists():
            raise FileNotFoundError(image_path)

        image = Image.open(image_path).convert("RGB")

        image_tensor = (
            self.preprocess(image)
            .unsqueeze(0)
            .to(self.device)
        )

        with torch.no_grad():

            embedding = self.model.encode_image(image_tensor)

            embedding = embedding / embedding.norm(
                dim=-1,
                keepdim=True,
            )

        embedding = (
            embedding.cpu()
            .numpy()
            .astype(np.float32)
        )

        return embedding[0]