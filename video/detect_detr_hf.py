#!/usr/bin/env python
import argparse
import cv2
import torch
from transformers import DetrImageProcessor, DetrForObjectDetection


THRESH = 0.7


def put_boxes(frame, outputs, processor, model):
    target_sizes = torch.tensor([frame.shape[:2]])
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=THRESH)[0]
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [int(i) for i in box.tolist()]
        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        text = f"{model.config.id2label[label.item()]} {score:.2f}"   # <-- используем model
        cv2.putText(frame, text, (box[0], max(0, box[1]-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
    return frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--save", default="out.mp4")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
    model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50").to(device)
    model.eval()

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(args.save, fourcc, fps, (w, h))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        inputs = processor(images=frame, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        frame = put_boxes(frame, outputs, processor, model)
        out.write(frame)

    cap.release()
    out.release()
    print(f"Saved: {args.save}")


if __name__ == "__main__":
    main()

# python detect_detr_hf.py --video data/test_video_2s.mp4 --save result.mp4
