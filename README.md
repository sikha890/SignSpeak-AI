# SignSpeak-AI 🤟

### Real-Time Sign Language Gesture Recognition Using Python and Computer Vision

SignSpeak-AI is a computer vision and machine learning project that recognizes selected static hand gestures in real time using a webcam. It uses MediaPipe to extract hand landmarks, OpenCV for video capture and display, and a Random Forest classifier to predict gestures.

The current prototype recognizes three gesture classes: **A, B, and C**.

## 🎯 Project Objective

To build a real-time hand gesture recognition prototype that combines hand landmark detection, dataset collection, machine learning, and live prediction.

## ✨ Features

- **Real-time webcam capture** using OpenCV.
- **Hand landmark detection** using MediaPipe.
- **Landmark-based feature extraction** using 21 hand landmarks with x, y, and z coordinates (63 features per sample).
- **Gesture data collection** for A, B, and C.
- **Model training and evaluation** using a Random Forest classifier.
- **Live gesture prediction** with confidence display.
- **Prediction smoothing** to help stabilize displayed predictions.

## 🖥️ Live Demo

The screenshot below shows the prototype predicting gesture C with 90.0% confidence.

![SignSpeak-AI live gesture recognition](screenshots/live-gesture-recognition.png)

## 🛠️ Technologies Used

- **Python** — application logic and model training.
- **OpenCV** — webcam access and image display.
- **MediaPipe** — hand detection and landmark extraction.
- **scikit-learn** — Random Forest classification and evaluation.
- **Joblib** — saving and loading the trained model.
- **NumPy** — numerical data processing.

## 📁 Project Structure

```text
SignSpeak-AI/
├── data/
│   ├── raw/
│   │   └── landmarks.csv
│   └── processed/
├── docs/
├── models/
│   └── gesture_model.joblib
├── screenshots/
│   ├── hand-tracking.png
│   └── live-gesture-recognition.png
├── src/
│   ├── collect_data.py
│   ├── hand_tracking.py
│   ├── predict_live.py
│   └── train_model.py
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/sikha890/SignSpeak-AI.git
cd SignSpeak-AI
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate the environment

**Windows — Git Bash:**

```bash
source .venv/Scripts/activate
```

**Windows — PowerShell:**

```powershell
.\.venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## ▶️ How to Run

Run the commands from the project's root directory with the virtual environment activated.

### 1. Hand tracking

```bash
python src/hand_tracking.py
```

This starts webcam-based hand tracking and displays detected hand landmarks.

### 2. Collect gesture data

```bash
python src/collect_data.py
```

Follow the prompts and on-screen instructions to collect samples for the supported gesture classes. The collected landmark data is stored in `data/raw/landmarks.csv`.

### 3. Train the model

```bash
python src/train_model.py
```

This trains the Random Forest classifier, evaluates it on a held-out test split, and saves the trained model to `models/gesture_model.joblib`.

### 4. Run live gesture recognition

```bash
python src/predict_live.py
```

The application opens the webcam, detects hand landmarks, and displays the predicted gesture and confidence.

**Tip:** Press `q` to quit the webcam window.

## 📊 Dataset and Model Evaluation

The current dataset and training run produced the following results:

| Metric | Result |
|---|---:|
| Collected samples | 1,112 |
| Input features per sample | 63 |
| Gesture classes | 3 (A, B, C) |
| Training samples | 889 |
| Test samples | 223 |
| Baseline test accuracy | 99.55% |

The model uses 21 hand landmarks, each represented by x, y, and z coordinates. These values form the 63 input features used by the classifier.

**Evaluation note:** The reported accuracy comes from a random train/test split of the collected dataset. If samples from the same recording sessions appear in both sets, this can overestimate performance on unseen users or recording conditions. Further testing on independent recordings is needed to assess real-world generalization.

## 📌 Current Scope and Limitations

- The current prototype recognizes static gestures for A, B, and C only.
- It is not a complete sign-language translation system.
- Recognition can be affected by lighting, camera quality, hand orientation, and background conditions.
- Additional gesture classes and independent evaluation data are needed to expand and validate the system.

## 🚀 Future Improvements

- Add more static gesture classes.
- Collect a larger and more diverse dataset.
- Evaluate the model on independent recording sessions and unseen users.
- Improve robustness under different lighting and hand positions.
- Explore dynamic gesture recognition and sentence-level translation as future extensions.

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## 👩‍💻 Author

**Sikha Kumari**

GitHub: [sikha890](https://github.com/sikha890)
