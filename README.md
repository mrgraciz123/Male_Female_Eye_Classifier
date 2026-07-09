# 👁️ Eye Gender Classifier Web Application

This repository contains a Convolutional Neural Network (CNN) designed in TensorFlow to classify eye photographs as belonging to either **Male** or **Female** eyes. The project features an interactive, modern web application built using **Streamlit** for local execution and deployment.

---

## 📁 Repository Structure

*   **`detection_of_male_and_female_eyes_from_image_datset.py`**: The training script that fetches the Kaggle dataset, trains the CNN model, prints accuracy and loss metrics, and exports the final model as `model.keras`.
*   **`app.py`**: The Streamlit application containing a high-fidelity user interface, custom styling, image preprocessing pipeline, confidence visualization, and an insights panel.
*   **`requirements.txt`**: The lists of dependencies required for compiling the model and running the Streamlit server.
*   **`README.md`**: Guide outlining project setup and usage.

---

## ⚙️ Prerequisites & Setup

Ensure you have Python 3.8+ installed on your local machine.

### 1. Install Dependencies
Install all the required python packages using `pip`:
```bash
pip install -r requirements.txt
```

### 2. Prepare the Model (`model.keras`)
To classify images, the Streamlit app requires a trained model named `model.keras` in the root folder. You have three ways to obtain it:

1.  **Train the model locally**:
    Run the training script (requires Kaggle API keys configured or datasets placed in the appropriate directory):
    ```bash
    python detection_of_male_and_female_eyes_from_image_datset.py
    ```
2.  **Generate a mock/baseline model**:
    If you want to quickly test the application's interface and pipeline immediately, start the Streamlit app. The interface detects if the model file is missing and offers an **"Initialize Baseline Model"** button to build the CNN architecture with random weights.
3.  **Upload custom weights**:
    You can upload your own pre-trained `.keras` or `.h5` files directly through the Control Panel in the Streamlit sidebar.

---

## 🚀 Running the Streamlit App

Launch the Streamlit web application on your local machine:
```bash
streamlit run app.py
```
This will start a local server and print the address (usually `http://localhost:8501`) in your terminal. Open this URL in any web browser to interact with the application.

---

## 🧠 Model Architecture details

The classification pipeline uses a custom Sequential CNN:
*   **Input Layer**: Accepts RGB images resized to `(299, 299, 3)` normalized between `[0, 1]`.
*   **Convolutional Blocks**: 4 blocks of Conv2D + MaxPooling layers mapping feature dimensions from 32 to 128 channels.
*   **Dense Head**: Flatten layer followed by a Dense layer of 128 units (ReLU) and a final sigmoid output neuron for binary prediction.
*   **Classes**:
    *   **`0` (probability ≤ 0.5)**: Female Eye
    *   **`1` (probability > 0.5)**: Male Eye

---

## 📊 Dataset Reference
The dataset used for training is the **Eyes Dataset (RTTE)** hosted on Kaggle, containing ~11,000 eye photograph samples categorized by class.
*   **Dataset URL**: [Kaggle Eyes RTTE Dataset](https://www.kaggle.com/datasets/pavelbiz/eyes-rtte)
