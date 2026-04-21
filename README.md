# 🏠 Bangalore House Price Prediction

A comprehensive end-to-end Machine Learning project that predicts real estate prices in Bangalore, India. This application leverages a Linear Regression model and features a modern, responsive web interface built with Flask.

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white)
![Linear Regression](https://img.shields.io/badge/Linear%20Regression-00599C?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)

## ✨ Key Features

- **Linear Regression Modeling**: Built using **Linear Regression** for reliable and interpretable price predictions.
- **Robust Data Pipeline**: Comprehensive data cleaning, handling missing values, and advanced outlier removal (price-per-sqft and BHK-specific).
- **Glassmorphism UI**: A premium, modern web interface featuring smooth animations and a responsive design.
- **Location-Aware Predictions**: Supports hundreds of Bangalore localities with intelligent 'other' categorization for rare locations.
- **End-to-End Workflow**: Includes scripts for data preprocessing, model training, and web deployment.

## 🚀 Getting Started

### Prerequisites

Ensure you have Python 3.8+ installed. You can install the required dependencies using:

```bash
pip install -r requirements.txt
```

### 1. Model Training
To train the model from scratch using the provided `data.csv`:

```bash
python train_model.py
```
*This will generate `model/bangalore_house_price_model.pkl`, `model/metadata.pkl`, and `model/locations.json`.*

### 2. Running the Web App
Start the Flask development server:

```bash
python app.py
```
Visit `http://127.0.0.1:5000` in your browser to start predicting!

## 📁 Project Structure

```text
Bangalore_House_price/
├── app.py              # Flask application & Prediction API
├── train_model.py      # ML pipeline: Cleaning, Training & Export
├── data.csv            # Raw dataset
├── requirements.txt    # Project dependencies
├── model/              # Serialized models and metadata
│   ├── bangalore_house_price_model.pkl
│   ├── metadata.pkl
│   └── locations.json
├── static/             # Frontend assets
│   └── style.css       # Glassmorphism styling
└── templates/          # HTML templates
    └── index.html      # Main UI
```

## 🧠 Machine Learning Insights

The model follows a rigorous preprocessing workflow:
1. **Data Cleaning**: Removal of irrelevant columns (area type, society, balcony).
2. **Feature Engineering**: Converting BHK and total square footage into numerical formats.
3. **Outlier Removal**:
   - Removing properties with unrealistic square footage per BHK (<300 sqft).
   - Statistical outlier removal based on mean and standard deviation of Price Per Sqft per location.
   - Removing BHK price anomalies (e.g., 2 BHK costing more than 3 BHK in the same area).
4. **Encoding**: One-Hot Encoding for locations with more than 10 data points.
5. **Model**: Linear Regression for robust baseline performance and interpretability.

## 🎨 UI Design
The application features a **Glassmorphism** design language:
- Semi-transparent glass containers.
- Vibrant background blobs with blur effects.
- Modern typography using the **Outfit** font.
- Responsive layout for mobile and desktop usage.

## 📄 License
This project is open-source and available under the [MIT License](LICENSE).

---
*Created with ❤️ for data science enthusiasts.*
