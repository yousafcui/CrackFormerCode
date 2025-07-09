# CrackFormerCode
CrackFormer: An Electronics-Driven Reinforcement Learning-Enhanced Swin Transformer V3 Framework for Communication-Efficient Pavement Crack Detection within IoT Network 
# CrackFormer: Reinforcement Learning-Enhanced Swin Transformer Framework for Pavement Crack Detection

CrackFormer is an advanced deep learning framework for automatic pavement crack detection and segmentation. It combines a **Mask R-CNN** for extracting region proposals, a **Swin Transformer V3** for learning multi-scale visual features, and an **Actor-Critic Reinforcement Learning agent** for optimizing crack boundary refinement. The model is designed for high segmentation fidelity and can be deployed in real-time on both GPUs and edge devices like Jetson Nano.

---

## 📂 Datasets

The following datasets are automatically downloaded using the Kaggle API:

| Dataset        | Description                                           | Kaggle Link |
|----------------|-------------------------------------------------------|-------------|
| Crack500       | Urban road surface crack images with fine labels      | [🔗 Link](https://www.kaggle.com/datasets/pauldavid22/crack50020220509t090436z001) |
| UAV-Crack      | Aerial UAV imagery of pavement cracks                 | [🔗 Link](https://www.kaggle.com/datasets/ziya07/uav-based-crack-detection-dataset) |
| SC-Crack       | Shadowed and rural crack textures                     | [🔗 Link](https://www.kaggle.com/datasets/lakshaymiddha/crack-segmentation-dataset) |

These will be downloaded automatically during the first run.

---

## 📦 Installation & Setup

### ✅ Dependencies

Install Python packages:

```bash
pip install torch torchvision timm Pillow scikit-learn opencv-python kaggle
