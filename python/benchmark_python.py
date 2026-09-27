#!/usr/bin/env python3
"""Benchmark the original Python/OpenCV style pipeline on the same synthetic input.

This intentionally excludes BPU inference because hobot_dnn is only available on
the RDK X5. It measures the camera-frame preprocessing, cable mask extraction,
synthetic coarse detection, and ROI refinement that can be compared directly
with cable_benchmark.
"""

import argparse
import json
import time

import cv2
import numpy as np


def synthetic_frame(width, height, index):
    image = np.full((height, width, 3), 220, np.uint8)
    cv2.line(image, (20, height // 2 - 35), (width - 20, height // 2 + 20), (28, 28, 28), 45, cv2.LINE_AA)
    cv2.line(image, (20, height // 2 - 35), (width - 20, height // 2 + 20), (55, 55, 55), 2, cv2.LINE_AA)
    cx = width // 2 + int(np.sin(index * 0.07) * width * 0.12)
    cy = height // 2 - 8 + int(np.cos(index * 0.05) * 4)
    box = cv2.boxPoints(((float(cx), float(cy)), (48.0, 24.0), -8.0)).astype(np.int32)
    cv2.fillConvexPoly(image, box, (210, 210, 210))
    cv2.polylines(image, [box], True, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.circle(image, (cx - 9, cy + 2), 5, (245, 245, 245), -1, cv2.LINE_AA)
    return image


def odd_kernel(value, minimum, maximum):
    value = int(np.clip(value, minimum, maximum))
    return value if value % 2 else value + 1


def fit_band(dark, extra_pad=10):
    h, w = dark.shape[:2]
    contours, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 0.002 * h * w:
            continue
        rw, rh = cv2.minAreaRect(contour)[1]
        if rw < 2 or rh < 2:
            continue
        aspect = max(rw, rh) / max(min(rw, rh), 1.0)
        _, _, bw, bh = cv2.boundingRect(contour)
        span = max(bw / max(w, 1), bh / max(h, 1))
        candidates.append((area * (1 + min(aspect, 14)) * (0.6 + span), contour))
    if not candidates:
        return None
    contour = max(candidates, key=lambda item: item[0])[1]
    if len(contour) < 8:
        return None
    vx, vy, x0, y0 = cv2.fitLine(contour, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
    direction = np.array([vx, vy], np.float32)
    direction /= max(float(np.linalg.norm(direction)), 1e-6)
    normal = np.array([-direction[1], direction[0]], np.float32)
    origin = np.array([x0, y0], np.float32)
    points = contour.reshape(-1, 2).astype(np.float32)
    rel = points - origin
    t = rel @ direction
    dist = np.abs(rel @ normal)
    t_min, t_max = np.percentile(t, [2, 98])
    if t_max - t_min < min(w, h) * 0.25:
        return None
    half_width = int(np.clip(np.percentile(dist, 82) + extra_pad, max(8, h * 0.05), max(12, h * 0.12)))
    extend = max(12, int(min(w, h) * 0.06))
    p1, p2 = origin + direction * (t_min - extend), origin + direction * (t_max + extend)
    polygon = np.array([p1 + normal * half_width, p2 + normal * half_width,
                        p2 - normal * half_width, p1 - normal * half_width], np.int32)
    polygon[:, 0] = np.clip(polygon[:, 0], 0, w - 1)
    polygon[:, 1] = np.clip(polygon[:, 1], 0, h - 1)
    mask = np.zeros((h, w), np.uint8)
    cv2.fillConvexPoly(mask, polygon, 255)
    k = odd_kernel(min(h, w) * 0.04, 3, 13)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))


def cable_mask(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    dark = np.where((lab[:, :, 0] < 90) | (hsv[:, :, 2] < 90), 255, 0).astype(np.uint8)
    dark = cv2.medianBlur(dark, 5)
    close_w = odd_kernel(frame.shape[1] * 0.18, 9, 61)
    close_h = odd_kernel(frame.shape[0] * 0.06, 3, 17)
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (close_w, close_h)), iterations=2)
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    fitted = fit_band(dark)
    if fitted is not None and 0.02 <= np.count_nonzero(fitted) / fitted.size <= 0.70:
        return fitted
    return np.ones(dark.shape, np.uint8) * 255


def normalize(data):
    data = data.astype(np.float32)
    if float(data.max() - data.min()) < 1e-6:
        return np.zeros_like(data)
    return cv2.normalize(data, None, 0, 1, cv2.NORM_MINMAX)


def refine(frame, bbox, mask, pad=24):
    x1, y1, x2, y2 = bbox
    h, w = frame.shape[:2]
    rx1, ry1, rx2, ry2 = max(0, x1 - pad), max(0, y1 - pad), min(w, x2 + pad), min(h, y2 + pad)
    roi = frame[ry1:ry2, rx1:rx2]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, roi_mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    k = odd_kernel(min(gray.shape[:2]) * 0.12, 5, 17)
    roi_mask = cv2.morphologyEx(roi_mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)), iterations=2)
    contours, _ = cv2.findContours(roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours or cv2.contourArea(max(contours, key=cv2.contourArea)) < max(20, 0.01 * gray.shape[0] * gray.shape[1]):
        roi_mask = mask[ry1:ry2, rx1:rx2]
    if roi_mask is None or not np.count_nonzero(roi_mask):
        return bbox
    blackhat_k = odd_kernel(min(gray.shape[:2]) * 0.16, 7, 21)
    blackhat = cv2.morphologyEx(blur, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (blackhat_k, blackhat_k)))
    gx = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.magnitude(gx, gy)
    local = cv2.GaussianBlur(blur, (15, 15), 0)
    contrast = cv2.absdiff(blur, local)
    score = ((0.5 * normalize(blackhat) + 0.3 * normalize(grad) + 0.2 * normalize(contrast)) * 255).clip(0, 255).astype(np.uint8)
    score = cv2.bitwise_and(score, score, mask=roi_mask)
    _, defects = cv2.threshold(score, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    defects = cv2.bitwise_and(defects, defects, mask=roi_mask)
    defects = cv2.morphologyEx(defects, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)), iterations=2)
    defects = cv2.morphologyEx(defects, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    contours, _ = cv2.findContours(defects, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid = [c for c in contours if cv2.contourArea(c) >= 12]
    if not valid:
        return bbox
    points = np.vstack(valid)
    rect = cv2.minAreaRect(points)
    box = cv2.boxPoints(rect).astype(np.int32)
    box += np.array([rx1, ry1])
    return [int(np.clip(box[:, 0].min(), 0, w)), int(np.clip(box[:, 1].min(), 0, h)),
            int(np.clip(box[:, 0].max(), 0, w)), int(np.clip(box[:, 1].max(), 0, h))]


def run(options):
    for _ in range(options.warmup):
        frame = synthetic_frame(options.width, options.height, 0)
        mask = cable_mask(frame)
        masked = frame.copy(); masked[mask == 0] = 127
        _ = cv2.resize(masked, (640, 480))
    times, detections = [], 0
    for i in range(options.frames):
        frame = synthetic_frame(options.width, options.height, i)
        start = time.perf_counter()
        mask = cable_mask(frame)
        masked = frame.copy(); masked[mask == 0] = 127
        scale = min(640 / frame.shape[0], 640 / frame.shape[1])
        nw, nh = int(frame.shape[1] * scale), int(frame.shape[0] * scale)
        padded = cv2.copyMakeBorder(cv2.resize(masked, (nw, nh)), (640 - nh) // 2, 640 - nh - (640 - nh) // 2,
                                    (640 - nw) // 2, 640 - nw - (640 - nw) // 2, cv2.BORDER_CONSTANT, value=127)
        gray = cv2.cvtColor(padded, cv2.COLOR_BGR2GRAY)
        _, bright = cv2.threshold(gray, 185, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) >= 12:
                x, y, bw, bh = cv2.boundingRect(contour)
                bbox = [int(np.clip((x - (640 - nw) / 2) / scale, 0, frame.shape[1])),
                        int(np.clip((y - (640 - nh) / 2) / scale, 0, frame.shape[0])),
                        int(np.clip((x + bw - (640 - nw) / 2) / scale, 0, frame.shape[1])),
                        int(np.clip((y + bh - (640 - nh) / 2) / scale, 0, frame.shape[0]))]
                refine(frame, bbox, mask)
                detections += 1
        times.append((time.perf_counter() - start) * 1000)
    report = {"backend": "python-opencv-synthetic", "frames": options.frames,
              "mean_ms": float(np.mean(times)), "p95_ms": float(np.percentile(times, 95)),
              "fps": float(1000 / np.mean(times)), "detections": detections}
    print(json.dumps(report, indent=2))
    if options.json:
        with open(options.json, "w", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--json", default="")
    run(parser.parse_args())
