# 🏎️ Formula 1 Race Predictor (2026 Ready)

A Machine Learning project that predicts Formula 1 podium probabilities. Built with **Python**, **Scikit-Learn**, and **FastF1**, this model is trained on race data from 2023-2025 to forecast the 2026 season.

## 🚀 Key Features
* **Multi-Season "Veteran" Model:** Trained on 3 years of data (2023-2025) to capture driver performance across different car generations.
* **2026 Season Ready:** Includes a custom translation layer to handle:
    * **New Teams:** Maps *Audi* to Sauber history and *Cadillac* to midfield proxies.
    * **Rookies:** Maps rookie *Arvid Lindblad* to teammate benchmarks (Liam Lawson) for realistic predictions.
    * **Rebrands:** Correctly distinguishes between *Red Bull Racing* (Verstappen) and *VCARB/RB* (Lawson).
* **Circuit-Specific Logic:** The model distinguishes between **Street Tracks** (e.g., Monaco, Singapore) and **Traditional Circuits** to account for specific car characteristics.

## 📊 Model Performance & Insights
The current V3 model (Random Forest Classifier) identified key trends in the 2024/2025 data:
* **The "Street Track" Bias:** The model correctly identifies that Red Bull's probability drops on street circuits due to ride-height issues, while McLaren's probability increases.
* **Qualifying Importance:** Grid position remains the single highest predictor of podium success (approx. 80% correlation).

## 🛠️ Installation & Usage

### 1. Clone the Repository
```bash
git clone https://github.com/GlenHan/F1_Predictor.git
cd F1_Predictor
```

2. Install Dependencies
```bash
pip install -r requirements.txt
```

```bash
3. Run the 2026 Predictor
python3 predict_2026.py
```

## 📂 File Structure
*`predict_2026.py`: The production script. Contains the dictionary maps for 2026 teams/drivers and runs the prediction engine.

*`f1_model_v3_multiyear.pkl`: The pre-trained Random Forest model (serialized).

*`f1_model_training_analysis.ipynb`: The research notebook containing data extraction, cleaning, feature engineering, and training visualizations.

*`f1_data_2023_2025_encoded.csv`: The processed dataset used for training.

## 🔮 Future Improvements
Weather Data integration: Fetching historical rain data to predict wet races.

Tyre Strategy: Incorporating compound choices (Soft/Medium/Hard) into the model.



Created by Glen Han

