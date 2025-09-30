#!/usr/bin/env python
import argparse
import json
import torch
from torchvision import models, transforms
from PIL import Image
import urllib.request

IMAGENET_LABELS_URL = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"


def load_labels():
    with urllib.request.urlopen(IMAGENET_LABELS_URL) as f:
        classes = [s.strip() for s in f.read().decode("utf-8").split("\n") if s.strip()]
    return classes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--topk", type=int, default=5)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT).to(device)
    model.eval()

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    img = Image.open(args.image).convert("RGB")
    x = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
    probs = torch.nn.functional.softmax(logits, dim=1)[0]

    classes = load_labels()
    topk = torch.topk(probs, k=args.topk)
    for p, idx in zip(topk.values.tolist(), topk.indices.tolist()):
        print(f"{classes[idx]}: {p:.4f}")


if __name__ == "__main__":
    main()

# python .\classify_torch.py --image .\data\img.png
