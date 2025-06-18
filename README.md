# 🌾 Rice Disease Detection using Deep Learning

<div align="center">

![Rice Disease Detection](https://img.shields.io/badge/Rice%20Disease-Detection-green?style=for-the-badge&logo=leaf&logoColor=white)
![Deep Learning](https://img.shields.io/badge/Deep-Learning-blue?style=for-the-badge&logo=tensorflow&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.8+-yellow?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=for-the-badge&logo=tensorflow&logoColor=white)

*An intelligent system for automated detection and classification of rice plant diseases using state-of-the-art deep learning models*

[📚 Documentation](#documentation) • [🚀 Quick Start](#quick-start) • [📊 Results](#results) • [🤝 Contributing](#contributing)

</div>

---

## 🎯 Project Overview

This project implements a comprehensive **Rice Disease Detection System** using advanced deep learning techniques. The system can automatically identify and classify four major rice diseases that significantly impact agricultural productivity:

- 🦠 **Bacterial Blight** - A destructive bacterial disease
- 💥 **Blast** - Fungal disease affecting leaves and panicles  
- 🟤 **Brown Spot** - Common fungal leaf spot disease
- 🦠 **Tungro** - Viral disease transmitted by leafhoppers

## ✨ Key Features

- 🔬 **Multiple Model Architecture**: Comparison of CNN, MobileNetV2, VGG16, and DenseNet121
- 📊 **High Accuracy**: Achieves excellent classification performance across all disease types
- 🔄 **Data Augmentation**: Advanced image preprocessing and augmentation techniques
- 📈 **Comprehensive Analysis**: Detailed performance metrics and visualization
- 🚀 **Transfer Learning**: Leverages pre-trained models for better performance
- ⚡ **Optimized Pipeline**: Efficient data loading and processing

## 🛠️ Tech Stack

<div align="center">

| Category | Technologies |
|----------|-------------|
| **ML/DL Framework** | ![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat&logo=tensorflow&logoColor=white) ![Keras](https://img.shields.io/badge/Keras-D00000?style=flat&logo=keras&logoColor=white) |
| **Data Processing** | ![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white) ![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white) ![OpenCV](https://img.shields.io/badge/OpenCV-27338e?style=flat&logo=opencv&logoColor=white) |
| **Visualization** | ![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c?style=flat) ![Seaborn](https://img.shields.io/badge/Seaborn-3776ab?style=flat) |
| **Environment** | ![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat&logo=jupyter&logoColor=white) ![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white) |

</div>

## 📋 Requirements

```python
tensorflow>=2.8.0
numpy>=1.21.0
matplotlib>=3.5.0
seaborn>=0.11.0
opencv-python>=4.5.0
pandas>=1.4.0
scikit-learn>=1.0.0
Pillow>=8.3.0
```

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/CHHEANG-UDOMVEASNA/Rice-Diseases-Detection.git
cd Rice-Diseases-Detection
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Dataset Setup
```bash
# Extract your dataset
python -c "
import zipfile
with zipfile.ZipFile('your_dataset.zip', 'r') as zip_ref:
    zip_ref.extractall('./dataset')
"
```

### 4. Run the Notebook
```bash
jupyter notebook mini_project.ipynb
```

## 📊 Model Performance

<div align="center">

### 🏆 Model Comparison

| Model | Test Accuracy | Parameters | Training Time |
|-------|--------------|------------|---------------|
| **DenseNet121** | **98.5%** ⭐ | 7.2M | ~45 min |
| **MobileNetV2** | 96.8% | 2.3M | ~30 min |
| **VGG16** | 95.2% | 134M | ~60 min |
| **Custom CNN** | 92.1% | 1.2M | ~25 min |

</div>

### 📈 Training Visualization

The project includes comprehensive visualizations:
- 📊 Training/Validation Loss & Accuracy curves
- 🎯 Confusion matrices for each model
- 📋 Detailed classification reports
- 🔍 Sample predictions with confidence scores

## 🔬 Model Architecture Details

### 🧠 DenseNet121 (Best Performing)
```python
- Base Model: DenseNet121 (ImageNet pretrained)
- Global Average Pooling
- Dense Layer: 256 units (ReLU)
- Dropout: 0.7
- Output Layer: 4 units (Softmax)
```

### 📱 MobileNetV2 (Most Efficient)
```python
- Base Model: MobileNetV2 (ImageNet pretrained)
- Global Average Pooling
- Dense Layer: 256 units (ReLU)
- Dropout: 0.7
- Output Layer: 4 units (Softmax)
```

## 📁 Project Structure

```
📦 Rice-Diseases-Detection/
├── 📓 mini_project.ipynb          # Main notebook
├── 📊 dataset/                    # Dataset directory
│   ├── 🦠 Bacterialblight/
│   ├── 💥 Blast/
│   ├── 🟤 Brownspot/
│   └── 🦠 Tungro/
├── 🤖 models/                     # Saved models
│   ├── densenet_model.h5
│   └── best_mobilenetv2_model.h5
├── 📋 requirements.txt            # Dependencies
├── 📖 README.md                   # This file
└── 📈 results/                    # Results and plots
```

## 🔧 Advanced Features

### 🎨 Data Augmentation Pipeline
- **Rotation**: ±40 degrees
- **Width/Height Shift**: 20%
- **Shear Transformation**: 20%
- **Zoom**: 20%
- **Horizontal Flip**: Random
- **Normalization**: [0,1] scaling

### ⚡ Optimization Techniques
- **Early Stopping**: Prevents overfitting
- **Learning Rate Scheduling**: Dynamic LR reduction
- **Model Checkpointing**: Saves best weights
- **Class Weights**: Handles imbalanced data

## 📊 Results & Analysis

### 🎯 Classification Report (DenseNet121)
```
                 precision    recall  f1-score   support
Bacterial Blight     0.99      0.98      0.98       238
Blast               0.98      0.99      0.98       216
Brown Spot          0.99      0.98      0.99       240
Tungro              0.97      0.98      0.98       196

accuracy                                0.985       890
macro avg           0.98      0.98      0.98       890
weighted avg        0.985     0.985     0.985      890
```

## 🚀 Usage Examples

### Single Image Prediction
```python
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('models/densenet_model.h5')

# Predict on new image
image = tf.keras.preprocessing.image.load_img('path/to/image.jpg', target_size=(224, 224))
prediction = model.predict(tf.expand_dims(image, 0))
class_names = ['Bacterial Blight', 'Blast', 'Brown Spot', 'Tungro']
predicted_class = class_names[tf.argmax(prediction[0])]
confidence = tf.reduce_max(prediction[0]) * 100

print(f"Predicted: {predicted_class} ({confidence:.2f}% confidence)")
```

## 🤝 Contributing

We welcome contributions! Here's how you can help:

1. 🍴 **Fork** the repository
2. 🌿 **Create** a feature branch (`git checkout -b feature/AmazingFeature`)
3. 💾 **Commit** your changes (`git commit -m 'Add some AmazingFeature'`)
4. 📤 **Push** to the branch (`git push origin feature/AmazingFeature`)
5. 🔄 **Open** a Pull Request

### 🐛 Found a Bug?
Please open an issue with:
- Bug description
- Steps to reproduce
- Expected vs actual behavior
- Screenshots (if applicable)

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- 🎓 **Dataset**: Rice disease dataset contributors
- 🧠 **Pre-trained Models**: TensorFlow/Keras team
- 📚 **Research Papers**: Various agricultural AI research
- 🌾 **Agricultural Experts**: For domain knowledge

## 📞 Contact

<div align="center">

**CHHEANG UDOMVEASNA**

[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/CHHEANG-UDOMVEASNA)

[![Email](https://img.shields.io/badge/Email-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:udomveasnachheang@gmail.com)

</div>

---

<div align="center">

### 🌟 If this project helped you, please give it a star! ⭐

**Made with ❤️ for the agricultural community**

</div>