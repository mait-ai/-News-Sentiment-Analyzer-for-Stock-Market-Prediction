# -News-Sentiment-Analyzer-for-Stock-Market-Prediction


This project analyzes the sentiment of financial news headlines using **FinBERT** embeddings and an **XGBoost** classifier.  
It classifies news into **Positive**, **Neutral**, or **Negative** sentiment categories to assist in **stock market prediction**.

---

## Features
- Uses `ProsusAI/finbert` for financial text embeddings.
- XGBoost for sentiment classification.
- Dataset: `mltrev23/financial-sentiment-analysis`.
- Shows prediction confidence and visual performance metrics.

---

## Methodology
1. **Data Preprocessing**  
   Clean and format text, remove missing values, and normalize sentiment labels.

2. **Embedding Generation**  
   Convert sentences into 768-dimensional embeddings using FinBERT.

3. **Model Training**  
   Train an XGBoost classifier on 400 samples and test on 100 samples.

4. **Evaluation**  
   Display accuracy, precision, recall, F1-score, and confusion matrix.

---

## Results
- **Accuracy:** ~76%
- **Balanced performance** across positive, neutral, and negative labels.
- Example:  
  *“Markets rise as investors gain confidence” → Predicted: Positive (0.93 confidence)*

---

## Future Work
- Incorporate **real-time financial APIs** for live news updates.
- Add **LSTM and transformer fine-tuning** for deeper contextual learning.
- Build a **dashboard UI** for dynamic sentiment trend visualization.
- Analyze **temporal market sentiment correlations**.

---

## How to Run
```bash
pip install transformers datasets pandas numpy tqdm scikit-learn xgboost
