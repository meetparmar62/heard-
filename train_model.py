import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
import pickle

# Create synthetic heart disease dataset (based on real UCI dataset structure)
np.random.seed(42)
n_samples = 1000

# Generate realistic medical data
data = {
    'Age': np.random.randint(29, 77, n_samples),
    'Sex': np.random.choice(['M', 'F'], n_samples),
    'ChestPainType': np.random.choice(['ATA', 'NAP', 'ASY', 'TA'], n_samples),
    'RestingBP': np.random.randint(80, 200, n_samples),
    'Cholesterol': np.random.randint(126, 500, n_samples),
    'FastingBS': np.random.choice([0, 1], n_samples),
    'RestingECG': np.random.choice(['Normal', 'ST', 'LVH'], n_samples),
    'MaxHR': np.random.randint(88, 202, n_samples),
    'ExerciseAngina': np.random.choice(['Y', 'N'], n_samples),
    'Oldpeak': np.round(np.random.uniform(-2, 6, n_samples), 1),
    'ST_Slope': np.random.choice(['Up', 'Flat', 'Down'], n_samples),
}

df = pd.DataFrame(data)

# Create target variable (realistic correlations)
target = np.zeros(n_samples)
for i in range(n_samples):
    risk_score = 0
    if df.loc[i, 'Age'] > 50:
        risk_score += 2
    if df.loc[i, 'Sex'] == 'M':
        risk_score += 1
    if df.loc[i, 'ChestPainType'] in ['ASY', 'TA']:
        risk_score += 2
    if df.loc[i, 'RestingBP'] > 140:
        risk_score += 1
    if df.loc[i, 'Cholesterol'] > 240:
        risk_score += 1
    if df.loc[i, 'FastingBS'] == 1:
        risk_score += 1
    if df.loc[i, 'MaxHR'] < 130:
        risk_score += 1
    if df.loc[i, 'ExerciseAngina'] == 'Y':
        risk_score += 2
    if df.loc[i, 'Oldpeak'] > 2:
        risk_score += 2
    if df.loc[i, 'ST_Slope'] == 'Down':
        risk_score += 2
    
    # Add some randomness
    risk_score += np.random.uniform(-2, 2)
    target[i] = 1 if risk_score > 6 else 0

df['HeartDisease'] = target.astype(int)

print("Dataset created:")
print(f"Total samples: {len(df)}")
print(f"Heart Disease cases: {df['HeartDisease'].sum()}")
print(f"Normal cases: {len(df) - df['HeartDisease'].sum()}")
print(f"\nClass distribution: {df['HeartDisease'].value_counts().to_dict()}")

# Separate features and target
X = df.drop('HeartDisease', axis=1)
y = df['HeartDisease']

# Define categorical and numerical columns
categorical_cols = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
numerical_cols = ['Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak']

# Create preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_cols)
    ]
)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Create pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42))
])

# Train model
print("\nTraining Random Forest model...")
pipeline.fit(X_train, y_train)

# Evaluate
y_pred = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy * 100:.2f}%")
print(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")

# Save preprocessor separately
with open('preprocessor.pkl', 'wb') as f:
    # Extract just the preprocessor from the pipeline
    pickle.dump(pipeline.named_steps['preprocessor'], f)
print("\nPreprocessor saved as 'preprocessor.pkl'")

# Save the entire pipeline as the model
with open('random_forest_model.pkl', 'wb') as f:
    pickle.dump(pipeline, f)
print("Model saved as 'random_forest_model.pkl'")

print("\n✅ Model training complete!")
