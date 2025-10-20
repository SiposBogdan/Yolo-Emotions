# 😃 Emotion Detection using YOLOv8

This repository contains a Jupyter Notebook (`sample.ipynb`) and supporting files that implement **facial emotion detection** using the **YOLOv8** deep learning model.  
The goal is to classify human facial expressions into **eight emotion categories** directly from images or videos.

---

## 💡 Motivation & Context

Human–machine interaction is becoming increasingly common through smartphones, virtual assistants, and social robots.  
Understanding emotional states improves user experience, enabling **adaptive and empathetic responses**.

Automatic facial emotion detection has wide applications:

- 🧠 **Psychology & Medicine:** Support diagnosis and monitoring of emotional disorders (e.g., depression, anxiety).  
- 💬 **Human–Computer Interaction:** Enhance natural and emotionally aware communication.  
- 📈 **Marketing & AR:** Enable emotion-based content personalization and engagement analysis.

Facial expressions are powerful **non-verbal indicators** of internal states, yet recognizing them in real-world (“in-the-wild”) conditions is challenging due to variations in lighting, pose, and image quality.  
This project addresses these challenges using **YOLOv8**, optimized for both accuracy and efficiency.

---

## 📊 Dataset — *AffectNet (YOLO format)*

We used the **AffectNet** dataset, one of the largest and most diverse emotion datasets available:

- **Scale:** 420,000+ manually labeled images covering diverse faces, ages, ethnicities, and conditions.  
- **Classes (8):**  
  `anger`, `contempt`, `disgust`, `fear`, `happy`, `neutral`, `sad`, `surprise`
- **Format:** Converted into YOLOv8 structure with 96×96 px images and normalized bounding-box coordinates.
- **Balanced sampling:** 800 training and 200 validation examples per class to avoid class imbalance.

## ⚙️ Model & Training Setup

### Model
- Base: `yolov8m.pt` (medium variant pretrained on COCO)
- Image size: `96×96`
- Epochs: `200`
- Batch size: `8`
- Patience (early stopping): `50`
- Optimized for limited GPU VRAM with full augmentation pipeline.

### Augmentations
- **Mosaic (0.5)** – mix 4 images to improve context robustness  
- **MixUp (0.2)** – combine pairs of images to regularize class boundaries  
- **HSV adjustments:** hue ±1.5%, saturation ±70%, value ±40%  
- **Flips:** 20% vertical (`flipud`), 50% horizontal (`fliplr`)  

These augmentations make the model robust to lighting, color, and orientation variations.

---

## 🧠 Results Summary (per class)

| Emotion | mAP@0.5 | F1_max | Main Confusions | Notes |
|----------|----------|--------|-----------------|-------|
| 😄 **Happy** | **0.926** | **0.84** | Minor overlap with *Surprise* | Most consistent — open smile features |
| 😨 **Fear** | 0.796 | 0.77 | Confused with *Surprise* (~10%) | Wide eyes overlap |
| 😲 **Surprise** | 0.790 | 0.71 | Confused with *Fear*, *Happy* | Similar open-mouth cues |
| 😏 **Contempt** | 0.738 | 0.72 | *Neutral*, *Sad* | Subtle asymmetry, lighting sensitive |
| 😡 **Anger** | 0.712 | 0.71 | *Disgust*, *Sad* | Frown overlaps |
| 😐 **Neutral** | 0.710 | 0.66 | *Sad*, *Contempt* | Hardest to separate; minimal expression |
| 🤢 **Disgust** | 0.656 | 0.64 | *Anger*, *Sad* | Subtle cues, resolution dependent |
| 😢 **Sad** | 0.578 | 0.59 | *Neutral*, *Contempt* | Most difficult due to subtlety |

**Overall performance**
- Mean F1 ≈ 0.70 at confidence ≈ 0.5  
- Median precision ≈ 0.62, recall ≈ 0.78  
- Optimal confidence threshold: ~0.5 for balanced precision-recall

**Best performing emotions:** *Happy*, *Fear*, *Surprise*  
**Challenging emotions:** *Sad*, *Disgust*, *Neutral*

---
