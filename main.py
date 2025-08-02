import pandas as pd
import numpy as np
import pickle
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# Load dataset
df = pd.read_csv('Crop_recommendation.csv')

# Fill missing values
for col in ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']:
    df[col] = df[col].fillna(df[col].mean())
df['label'] = df['label'].fillna(0)

# Label encoding
le = preprocessing.LabelEncoder()
df['label'] = le.fit_transform(df['label'].astype(str))

# Save label mapping
label_mapping = dict(zip(df['label'], le.inverse_transform(df['label'])))
with open('label_mapping.pkl', 'wb') as f:
    pickle.dump(label_mapping, f)

# Prepare data
X = df.drop('label', axis=1)
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Save metrics
with open('metrics.txt', 'w') as f:
    f.write(f"Accuracy: {accuracy}\n")
    f.write(report)

# Save feature importance
importances = model.feature_importances_
features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
fi_df = pd.DataFrame({"Feature": features, "Importance": importances})
fi_df.to_csv("feature_importance.csv", index=False)

# Save model
with open('data.pkl', 'wb') as f:
    pickle.dump(model, f)

print("All files generated successfully.")
