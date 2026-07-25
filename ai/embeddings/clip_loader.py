import torch
import open_clip


class CLIPLoader:
    """
    Loads the OpenCLIP model and preprocessing transforms.
    """

    def __init__(
        self,
        model_name="ViT-B-32",
        pretrained="laion2b_s34b_b79k",
        device=None,
    ):

        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        print(f"Loading CLIP Model: {model_name}")
        print(f"Using Device: {self.device}")

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name=model_name,
            pretrained=pretrained,
            device=self.device,
        )

        self.tokenizer = open_clip.get_tokenizer(model_name)

        self.model.eval()

    def get_model(self):
        return self.model

    def get_preprocess(self):
        return self.preprocess

    def get_device(self):
        return self.device

    def get_tokenizer(self):
        return self.tokenizer