# 🏠 Ames Housing Price Predictor: Ultimate AI-Powered Real Estate Estimator

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg?style=for-the-badge&logo=python)
![Gradio](https://img.shields.io/badge/Gradio-UI-orange.svg?style=for-the-badge)
![ML](https://img.shields.io/badge/Machine%20Learning-20%20Models%20Compared-green.svg?style=for-the-badge)
![NLP](https://img.shields.io/badge/NLP-Gemini%20AI-purple.svg?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-red.svg?style=for-the-badge)

Welcome to the **Ames Housing Price Predictor**! This is a comprehensive, state-of-the-art AI-powered application designed to estimate real estate prices with exceptional accuracy. By seamlessly combining advanced **Machine Learning** algorithms with **Natural Language Processing (NLP)**, this tool provides an incredibly intuitive and powerful user experience.

## 📖 Repository Description

This repository contains the complete end-to-end machine learning pipeline and the interactive web application for predicting house prices based on the Ames Housing Dataset. The project demonstrates a rigorous data science workflow—from deep exploratory data analysis and extensive feature engineering to model training, evaluation, and finally, deployment.

### 🧠 The Quest for the Best Accuracy: 20 Models Compared
To ensure the highest possible predictive accuracy, we didn't just stop at one model. **We trained, tuned, and compared 20 different machine learning models!** After rigorous cross-validation and hyperparameter tuning, the **Extra Trees Regressor** emerged as the champion, delivering unparalleled precision, robust performance, and the best R² Score.

## � Key Features & Functionalities

### 💬 1. Intelligent Chat Mode (Natural Language)
Gone are the days of endless dropdowns and forms. Just describe your house naturally in plain English, and our NLP engine (powered by custom regex and Google's Gemini AI) will instantly parse your features and predict the price.
> *"I have a beautiful 3-bedroom house, around 2000 sq ft, built in 2005 with a spacious 2-car garage."*

### 🎚️ 2. High-Precision Slider Mode
For users who want granular control, the traditional interactive interface offers precise sliders and dropdowns to fine-tune every single feature of the property.

### 📊 3. Comparable Homes Analysis
Don't just get a number—understand the market. The application fetches and displays **similar properties** from the historical dataset so you can visually compare features and actual sale prices.

### 💡 4. AI-Powered Price Optimization Tips
Want to increase your home's value? The application generates actionable, AI-driven recommendations tailored specifically to your property's characteristics, suggesting renovations or improvements that yield the highest return on investment.

## 📈 Champion Model Performance

Our rigorous evaluation yielded exceptional results for the final deployed model:

| Metric | Champion Model | Performance |
|--------|----------------|-------------|
| **R² Score** | Extra Trees Regressor | **0.914** |
| **RMSE** | Extra Trees Regressor | **$21,773** |
| **Features Used**| Engineered Dataset | **164 Features** |

## 🚀 Quick Start & Installation

```bash
# Clone the repository
git clone https://github.com/your-username/house-price-estimator.git
cd house-price-estimator

# Install dependencies
pip install gradio pandas numpy scikit-learn google-generativeai

# Run the amazing app
python3 app.py
```
*Open http://127.0.0.1:7862 in your browser to experience the magic.*

## 📁 Detailed Project Structure

```
├── app.py                 # Core Gradio UI & Inference Application
├── notebooks/             # Data Science Workflow Core
│   ├── 01_preprocessing   # Advanced data cleaning & sophisticated feature engineering
│   ├── 02_eda             # Deep Exploratory Data Analysis & visual insights
│   ├── 03_modeling        # Training and rigorous comparison of 20 ML models
│   ├── 04_evaluation      # Final model selection & performance metrics
│   └── 05_app             # Code prototyping for the resulting web application
├── data/                  # Datasets Repository
│   ├── train.csv          # Original Ames Housing dataset
│   └── data_preprocessed  # Engineered dataset ready for inference
└── models/                # Saved Model Artifacts
    ├── final_model.pkl    # Serialized Extra Trees champion model
    └── preprocessors.pkl  # Pipelines, robust scalers, and custom encoders
```

## �️ Technologies Stack

This project leverages a modern, robust, and highly scalable technology stack:
- **Core Language**: Python 3.9+ 🐍
- **Machine Learning Framework**: `scikit-learn` (for model building, pipelines, and evaluation)
- **Data Manipulation**: `Pandas`, `NumPy`
- **Interactive Web UI**: `Gradio` (for real-time, user-friendly frontend)
- **Natural Language Processing**: Custom Regex Engine & `Google Generative AI`
- **Dataset**: Comprehensive Ames Housing Data

## 📝 License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for more details. Feel free to fork, use, and modify!

---
*Built with ❤️ using Python, Advanced Machine Learning, and cutting-edge NLP.*
