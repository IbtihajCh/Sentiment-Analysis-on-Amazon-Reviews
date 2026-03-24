# Amazon Electronics Reviews Sentiment Analysis

## Overview
This project performs sentiment classification on **50,000 Amazon Electronics product reviews**, sampled from a larger dataset of 500,000 records. The goal is to categorize reviews into **Positive, Neutral, and Negative** sentiments using a **CNN-LSTM model**, handling class imbalance and optimizing performance.

The repository includes the methodology, data analysis, model development, evaluation, and findings.

---

## Dataset

- **Original Size:** 1193.88 MB (500,000 reviews)  
- **Sampled Records:** 50,000  
- **Columns:** `reviewText`, `overall`  
- **Memory Usage:** ~1.1 MB  

### Sample Reviews
| Rating | Review Excerpt |
|--------|----------------|
| 5.0    | "It is small, it is inexpensive and best of all..." |
| 4.0    | "This was my first GPS and I was pleasantly sur..." |
| 5.0    | "Great way to cobble together all my shorter co..." |
| 2.0    | "I bought this on New Years 2006 and I used it ..." |
| 5.0    | "I've been a fan of high quality headphones for..." |

---

## Exploratory Data Analysis (EDA)

- **Class Distribution (Before Balancing):**  
  - Positive: 41,313  
  - Negative: 5,122  
  - Neutral: 3,565  

- **Balanced Dataset:** Each class adjusted to 16,666 samples  
- **Review Length:** Mean = 72.84 words, Max = 1,701 words  

Visualizations include rating distribution and review length histograms (capped at 200 words for clarity).

---

## Model Architecture

The CNN-LSTM model was structured as follows:

1. **Embedding Layer:** 20,000 vocabulary, 128 dimensions  
2. **Conv1D Layers:** 64 & 128 filters with BatchNormalization  
3. **MaxPooling1D & Dropout** layers  
4. **LSTM Layer:** 100 units with BatchNormalization  
5. **Dense Layers:** 128 → 64 → 3 outputs with Dropout  

- **Total Parameters:** 8,354,203  
- **Trainable Parameters:** 2,784,539  
- **Model Size:** 31.87 MB  

**Training:** 50 epochs with early stopping at epoch 17  
**Validation Accuracy:** 91.47%  
**Training Time:** 4,428.61 seconds  

---

## Model Evaluation

**Test Set:** 7,500 samples  

### Classification Report
| Class    | Precision | Recall | F1-Score |
|----------|-----------|--------|----------|
| Negative | 0.94      | 0.93   | 0.93     |
| Neutral  | 0.86      | 0.93   | 0.89     |
| Positive | 0.92      | 0.85   | 0.88     |
| **Accuracy** | **0.90** | - | - |
| **Macro Avg** | 0.91 | 0.90 | 0.90 |
| **Weighted Avg** | 0.91 | 0.90 | 0.90 |

**Sample Predictions**
- `"This product is absolutely amazing..."` → Positive (0.981)  
- `"The product is decent..."` → Neutral (0.998)  
- `"Terrible quality, broke after 2 days..."` → Negative (0.997)  

**Inference Time:** 0.1041 – 0.1946 seconds per sample  

---

## Conclusion

The **CNN-LSTM model achieved 90% accuracy** on the balanced 50,000-review dataset. It successfully classified sentiments despite initial imbalance.  

**Future Work:**
- Use larger datasets  
- Incorporate advanced techniques such as **attention mechanisms**  
- Experiment with other deep learning architectures for improved performance  

---
