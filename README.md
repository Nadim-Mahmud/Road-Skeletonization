# Deep Learning-Based Learned Skeletonization of Road Networks

## 🛠 Project Overview

This project develops a deep learning-based model to recover thin skeleton representations from thick, noisy road network images generated from OpenStreetMap (OSM) data.  
We handle ambiguities inherent in skeletonization while leveraging advanced techniques such as U-Net with residual blocks, iterative thinning, and multi-task learning.

This work is completed for **CSE627: Advance Machine Learning**, following project guidelines.

## 📂 Project Structure

```
road_skeletonization/
├── datasets/
│   └── road_skeleton_dataset.py       # Dataset and collate function
├── engine/
│   ├── trainer.py                      # Trainer class
│   └── tester.py                       # Tester class
├── models/
│   ├── skeleton_unet.py                # U-Net model with residual blocks
│   └── loss.py                         # Weighted Focal Loss
├── transforms/
│   └── transforms.py                   # Data augmentation transforms
├── utils/
│   └── dataloader.py                   # Data splitting and loader functions
├── config.py                           # Configurations (paths, hyperparameters)
├── train.py                            # Training script
├── test.py                             # Testing and evaluation script
├── README.md                           # Project description
└── requirements.txt                    # (optional) Dependencies list
```

## 🚀 Setup Instructions

1. Clone the repository and navigate into the project folder.

```bash
git clone https://github.com/Nadim-Mahmud/Road-Skeletonization
cd Road-Skeletonization
```

2. Install required Python packages:

```bash
pip install torch torchvision torchmetrics matplotlib pillow
```

(Optional) Create and activate a virtual environment before installing.

3. Update dataset path: 
   Edit `config.py` and set:

```python
root_path = 'thinning_data/data'
```

## 🏋️‍♂️ Training the Model

Run the training script:

```bash
python train.py
```

This will:

- Train the model on the training set
- Validate it on the validation set
- Automatically save the best model as `best_model.pth`

Training uses **early stopping** if validation performance does not improve.

## 🧪 Testing the Model

After training completes, run:

```bash
python test.py
```

This will:

- Evaluate the best model on the test set
- Print Accuracy, Precision, Recall, and F1 Score
- Save sample visualizations to `test_samples.png`

## 📚 Project Details

### 📥 Data Generation

- **OSM Data Extraction**: Selected a subset of road types from OSM and retrieved the corresponding data.
- **Rasterization**: Rendered roads into 256×256 grayscale images with thickness based on attributes (e.g., lane count, width).
- **Distortions**: Introduced realistic distortions (blur, noise) to simulate real-world imperfections.
- **Ground Truth**: Generated 1-pixel thick skeletons from road centerlines.

### 🧠 Model Approaches

- **Baseline**: U-Net architecture to predict thin skeleton from thick, noisy input.
- **Advanced Option** (choose one):
  - **Iterative Thinning**: Simulate gradual de-thickening.
  - **Multi-Task Learning**: Incorporate auxiliary tasks (e.g., distance maps).
  - **Training on Misaligned Data**: Handle skeleton misalignment with custom losses or post-processing.

### 🔍 Testing & Verification

- **Verification**: Conducted unit tests, visual inspection, and controlled experiments with known outcomes.
- **Correctness**: Ensured every pipeline component (data, network, metrics) works correctly through intermediate inspections.

### 📈 Evaluation

- **Qualitative Results**: Visual samples showing input, ground truth, and model predictions.
- **Quantitative Metrics**:
  - Test Loss
  - Mean Squared Error (MSE) using distance transforms
  - Node Precision and Recall for 1-valent to 4-valent nodes
  - (Optional) IoU, Dice Coefficient

### 🧪 Ablation Study

Performed ablation study by varying:

- Learning rate
- Loss functions
- Architectural choices

Reported results in tables and justified the final chosen configuration.

## ⚙️ Implementation Requirements

- Project organized as **Python scripts** runnable from command line or Google Colab via `%run script.py`.
- GPU support compatible.
- Code must be linked but is **not graded** — only the **slides** are graded.
- Slides must display final qualitative/quantitative results, methodology, and ablation studies.

## 📈 Metrics and Visualization

Metrics tracked:

- Accuracy
- Precision
- Recall
- F1 Score
- MSE
- Node-based precision and recall

Visualization:

- Side-by-side comparisons of Input, Ground Truth, and Prediction images saved in `test_samples.png`.

