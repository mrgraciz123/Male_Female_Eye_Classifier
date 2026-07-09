# 👁️ CNN Eye Gender Classification Console

[![Streamlit App](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)](https://malefemaleeyeclassifier-8gfzkhji6mdg8nxp7jum2b.streamlit.app/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=TensorFlow&logoColor=white)](https://www.tensorflow.org/)
[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/abhayshankertiwari)

An advanced, interactive deep learning web console built using **Streamlit** and **TensorFlow** to classify eye crop photographs as **Male** or **Female**. Leveraging a custom 11-layer Convolutional Neural Network (CNN), this dashboard provides real-time decision telemetry, explainable Grad-CAM feature attention maps, and comprehensive analytics.

🌐 **Live Demo URL:** [https://malefemaleeyeclassifier-8gfzkhji6mdg8nxp7jum2b.streamlit.app/](https://malefemaleeyeclassifier-8gfzkhji6mdg8nxp7jum2b.streamlit.app/)

---

## ✨ Features

- 🛸 **Interactive SaaS Dark Theme:** Custom premium CSS injection with glassmorphism layout cards and holographic scanning laser animations.
- 🔮 **Dual Prediction Pipeline:** Standard file upload (`JPEG`, `PNG`) or direct webcam scan capture.
- 📊 **Decision Telemetry Console:** Multi-dimensional prediction stats showing real-time probability gauge indicators and interactive Plotly distribution charts.
- 🔬 **Explainable AI (XAI):** Neural Focus maps using simulated Grad-CAM to highlight high-contrast features (pupil, iris, lashes) triggering the classification.
- 📈 **Interactive Model Analytics:** Responsive validation charts mapping accuracy progression and validation loss convergence across training epochs.
- ⚙️ **Model Architecture Specifications:** Layer-by-layer parameter configuration logs mapping the deep learning network.
- 📜 **Historical Session Database:** Cached local session history with timestamps, thumbnails, and prediction results.
- 🛡️ **Sandbox Fallback Mode:** Operates automatically in brightness-contrast simulation sandbox if TensorFlow weights are unmounted.

---

## 📂 Project Structure

```bash
├── Detection_of_Male_and_Female_Eyes_from_image_Datset.ipynb  # Jupyter Notebook containing dataset pipeline & training logs
├── app.py                                                    # Premium Streamlit web console dashboard application
├── eye_gender_model.keras                                    # Pre-trained CNN model file containing model weights (~51.6 MB)
├── requirements.txt                                          # Server configuration dependencies
└── README.md                                                 # Developer documentation
```

---

## 🚀 Local Installation & Execution

### 1. Clone the Repository
```bash
git clone https://github.com/mrgraciz123/Male_Female_Eye_Classifier.git
cd Male_Female_Eye_Classifier
```

### 2. Set Up Virtual Environment (Recommended)
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On Linux/macOS
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit Application
```bash
streamlit run app.py
```
After execution, open `http://localhost:8501` in your web browser.

---

## 🧠 Neural Network Specifications

The CNN model architecture contains **4 Convolutional blocks** followed by a fully-connected Dense classifier:

| Layer | Type | Specifications | Output Shape | Activation | Parameters |
| :---: | :--- | :--- | :--- | :---: | :---: |
| **1** | Conv2D | 32 Filters (3x3 Kernel) | `(None, 297, 297, 32)` | ReLU | 896 |
| **2** | MaxPooling2D | Pool Size (2x2) | `(None, 148, 148, 32)` | N/A | 0 |
| **3** | Conv2D | 32 Filters (3x3 Kernel) | `(None, 146, 146, 32)` | ReLU | 9,248 |
| **4** | MaxPooling2D | Pool Size (2x2) | `(None, 73, 73, 32)` | N/A | 0 |
| **5** | Conv2D | 64 Filters (3x3 Kernel) | `(None, 71, 71, 64)` | ReLU | 18,496 |
| **6** | MaxPooling2D | Pool Size (2x2) | `(None, 35, 35, 64)` | N/A | 0 |
| **7** | Conv2D | 128 Filters (3x3 Kernel) | `(None, 33, 33, 128)` | ReLU | 73,856 |
| **8** | MaxPooling2D | Pool Size (2x2) | `(None, 16, 16, 128)` | N/A | 0 |
| **9** | Flatten | Flatten Matrix Array | `(None, 32768)` | N/A | 0 |
| **10** | Dense | 128 Fully Connected Nodes | `(None, 128)` | ReLU | 4,194,432 |
| **11** | Dense (Output) | 1 Binary Classification Node | `(None, 1)` | Sigmoid | 129 |

- **Total Parameters:** `4,297,057`
- **Trainable Parameters:** `4,297,057`
- **Binary Decision Threshold:** `0.5` (Female: `≤ 0.5`, Male: `> 0.5`)

---

## 📊 Dataset Insight
The network is trained on the public **Eyes Dataset (RTTE)** hosted on Kaggle:
- **Total Images:** 11,525 (80% training / 20% validation split)
- **Image Preprocessing:** Targets resized to `299x299` pixels, normalized to `[0, 1]`.
- **Augmentation Details:** 20° Rotations, Horizontal flips, 0.2 shear/zoom, and width/height shifts.
- **Dataset Link:** [Kaggle Eyes RTTE Dataset](https://www.kaggle.com/datasets/pavelbiz/eyes-rtte)

---

## 👤 Developer Profile

**Abhay Shanker Tiwari** — AI & ML Engineer
- **LinkedIn:** [abhayshankertiwari](https://www.linkedin.com/in/abhayshankertiwari)
- **GitHub:** [mrgraciz123](https://github.com/mrgraciz123)
- **Email:** [abhaylibra15@gmail.com](mailto:abhaylibra15@gmail.com)

Feel free to open issues or make pull requests to improve model metrics!
