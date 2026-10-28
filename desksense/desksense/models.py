import torch
from torch import nn


class VideoClassifier(nn.Module):
    def __init__(self, spec, pretrained=True):
        super().__init__()
        self.spec = spec
        kind = spec["model"]
        if kind == "videomae":
            from transformers import VideoMAEConfig, VideoMAEForVideoClassification
            config = VideoMAEConfig.from_dict(spec["hf_config"])
            if pretrained:
                self.network = VideoMAEForVideoClassification.from_pretrained(
                    spec["checkpoint"], revision=spec["revision"], config=config,
                    ignore_mismatched_sizes=True)
            else:
                self.network = VideoMAEForVideoClassification(config)
        elif kind == "frame_mean":
            from torchvision.models import ResNet18_Weights, resnet18
            self.network = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
            self.network.fc = nn.Linear(self.network.fc.in_features, len(spec["classes"]))
        elif kind == "tiny":
            # Synthetic CI only. Never a candidate for reported real-dataset results.
            self.network = nn.Sequential(nn.Conv3d(3, 8, 3, padding=1), nn.ReLU(),
                                         nn.AdaptiveAvgPool3d(1), nn.Flatten(),
                                         nn.Linear(8, len(spec["classes"])))
        else:
            raise ValueError(kind)

    def forward(self, x):
        # [B,T,C,H,W]
        if self.spec["model"] == "videomae":
            return self.network(pixel_values=x).logits
        if self.spec["model"] == "frame_mean":
            b, t, c, h, w = x.shape
            logits = self.network(x.reshape(b * t, c, h, w))
            return logits.reshape(b, t, -1).mean(1)
        return self.network(x.permute(0, 2, 1, 3, 4))

    def trainable(self, mode):
        if mode not in {"head", "partial", "full"}:
            raise ValueError(mode)
        self.mode = mode
        for p in self.parameters():
            p.requires_grad = mode == "full"
        if self.spec["model"] == "videomae":
            modules = [self.network.classifier, self.network.fc_norm]
            if mode == "partial":
                modules.extend(list(self.network.videomae.encoder.layer[-2:]))
        elif self.spec["model"] == "frame_mean":
            modules = [self.network.fc]
            if mode == "partial":
                modules.append(self.network.layer4)
        else:
            modules = [self.network]
        for module in modules:
            if module is None:
                continue
            for p in module.parameters():
                p.requires_grad = True
        return [p for p in self.parameters() if p.requires_grad]

    def train(self, mode=True):
        super().train(mode)
        # Freeze BN statistics and backbone dropout during head/partial training.
        if mode and getattr(self, "mode", "full") != "full":
            self.network.eval()
            if self.spec["model"] == "videomae":
                self.network.classifier.train()
                if self.mode == "partial":
                    for layer in self.network.videomae.encoder.layer[-2:]:
                        layer.train()
            elif self.spec["model"] == "frame_mean":
                self.network.fc.train()
                # Keep BN eval even in partially trained ResNet.
        return self


def make_spec(model, classes, checkpoint="MCG-NJU/videomae-base", revision="main"):
    spec = dict(model=model, classes=classes, frames=16, size=224,
                mean=[.485, .456, .406], std=[.229, .224, .225])
    if model == "videomae":
        from transformers import AutoImageProcessor, VideoMAEConfig
        config = VideoMAEConfig.from_pretrained(checkpoint, revision=revision)
        processor = AutoImageProcessor.from_pretrained(checkpoint, revision=revision, use_fast=False)
        # Standard fine-tuning policy: mean-pool all video tokens, then normalize.
        config.use_mean_pooling = True
        config.num_labels = len(classes)
        config.id2label = dict(enumerate(classes))
        config.label2id = {c: i for i, c in enumerate(classes)}
        # Resolve main to the immutable Hub commit when metadata supplies it.
        resolved = getattr(config, "_commit_hash", None) or revision
        spec.update(hf_config=config.to_dict(), checkpoint=checkpoint, revision=resolved,
                    frames=config.num_frames, size=config.image_size,
                    mean=processor.image_mean, std=processor.image_std)
    elif model == "tiny":
        spec.update(frames=4, size=32)
    return spec


def load_model(path, device):
    # Only load trusted project checkpoints. weights_only disallows arbitrary pickle code.
    bundle = torch.load(path, map_location="cpu", weights_only=True)
    model = VideoClassifier(bundle["spec"], pretrained=False)
    model.load_state_dict(bundle["state_dict"])
    return model.to(device).eval(), bundle
