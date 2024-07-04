Yes, you mentioned training a suitable anomaly detection model, such as Isolation Forest or One-Class SVM. However, in your code, only the Isolation Forest model is used. If you want to include the use of SVM for anomaly detection, here is an updated version of the README file reflecting that:

---

# Keyword Anomaly Detection Project

This project aims to detect anomalies in a dataset containing product titles and keywords using Isolation Forest and Support Vector Machine (SVM) models. The dataset is processed to encode categorical features, and BERT embeddings are used for keyword representation. The anomalies are then saved to a CSV file.

## Prerequisites

Before you begin, ensure you have met the following requirements:
- Python 3.6 or later
- The following Python packages:
  - `pandas`
  - `numpy`
  - `matplotlib`
  - `scikit-learn`
  - `transformers`
  - `torch`

You can install these packages using `pip`:
```bash
pip install pandas numpy matplotlib scikit-learn transformers torch
```

## Installation

1. Clone this repository:
```bash
git clone https://github.com/your-username/keyword-anomaly-detection.git
```

2. Navigate to the project directory:
```bash
cd keyword-anomaly-detection
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Load your dataset into a pandas DataFrame:
```python
import pandas as pd
df = pd.read_csv(".......csv")
```

2. Encode categorical columns (`PRODUCT_TITLE` and `Keyword Type`) into numeric values:
```python
from sklearn.preprocessing import LabelEncoder

label_encoders = {}
for column in ["PRODUCT_TITLE", "Keyword Type"]:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le
```

3. Encode the `Keyword` column using BERT embeddings:
```python
from transformers import BertTokenizer, BertModel
import torch
import numpy as np

model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

keyword_embeddings = []
max_length = 128

for keyword in df['Keyword']:
    keyword = keyword[:max_length]
    inputs = tokenizer.encode_plus(keyword, add_special_tokens=True, padding=True, max_length=max_length, truncation=True, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs)
        hidden_states = outputs.last_hidden_state
    embeddings = hidden_states.mean(dim=1).squeeze().numpy()
    keyword_embeddings.append(embeddings)

X = np.hstack([df[["PRODUCT_TITLE", "Keyword Type"]].values, np.vstack(keyword_embeddings)])
```

4. Train the Isolation Forest model for anomaly detection:
```python
from sklearn.ensemble import IsolationForest

clf_isolation_forest = IsolationForest(contamination=0.05, random_state=42)
clf_isolation_forest.fit(X)

df["anomaly_score_if"] = clf_isolation_forest.decision_function(X)
df["is_anomaly_if"] = clf_isolation_forest.predict(X)
```

5. Train the One-Class SVM model for anomaly detection (optional):
```python
from sklearn.svm import OneClassSVM

clf_svm = OneClassSVM(kernel='rbf', gamma='auto', nu=0.05)
clf_svm.fit(X)

df["anomaly_score_svm"] = clf_svm.decision_function(X)
df["is_anomaly_svm"] = clf_svm.predict(X)
```

6. Save anomalies detected by Isolation Forest to a CSV file:
```python
anomalies_if = df[df["is_anomaly_if"] == -1]
anomalies_if.to_csv("anomalies_isolation_forest.csv", index=False)
print("Anomalies saved to 'anomalies_isolation_forest.csv'.")
```

7. Save anomalies detected by One-Class SVM to a CSV file (optional):
```python
anomalies_svm = df[df["is_anomaly_svm"] == -1]
anomalies_svm.to_csv("anomalies_svm.csv", index=False)
print("Anomalies saved to 'anomalies_svm.csv'.")
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License.

---

This README file includes instructions for using both the Isolation Forest and One-Class SVM models for anomaly detection.
