import requests
import json

# Test the FastAPI backend
API_URL = "http://127.0.0.1:8000/predict/"

print("=" * 60)
print("Testing Heart Disease Prediction API")
print("=" * 60)

# Test Case 1: High Risk Patient
print("\n📋 Test Case 1: High Risk Patient")
print("-" * 60)
test_case_1 = {
    "Age": 65,
    "Sex": "M",
    "ChestPainType": "ASY",
    "RestingBP": 160,
    "Cholesterol": 280,
    "FastingBS": 1,
    "RestingECG": "LVH",
    "MaxHR": 110,
    "ExerciseAngina": "Y",
    "Oldpeak": 3.5,
    "ST_Slope": "Down"
}

print("Input:")
for key, value in test_case_1.items():
    print(f"  {key}: {value}")

try:
    response = requests.post(API_URL, json=test_case_1)
    if response.status_code == 200:
        result = response.json()
        prediction = result["prediction"]
        print(f"\n✅ API Response: {result}")
        if prediction == 0:
            print("🟢 Result: NORMAL - No heart disease detected")
        else:
            print("🔴 Result: HIGH RISK - Heart disease detected!")
    else:
        print(f"❌ Error: Status Code {response.status_code}")
except Exception as e:
    print(f"❌ Error: {e}")

# Test Case 2: Low Risk Patient
print("\n\n📋 Test Case 2: Low Risk Patient")
print("-" * 60)
test_case_2 = {
    "Age": 35,
    "Sex": "F",
    "ChestPainType": "NAP",
    "RestingBP": 120,
    "Cholesterol": 180,
    "FastingBS": 0,
    "RestingECG": "Normal",
    "MaxHR": 170,
    "ExerciseAngina": "N",
    "Oldpeak": 0.5,
    "ST_Slope": "Up"
}

print("Input:")
for key, value in test_case_2.items():
    print(f"  {key}: {value}")

try:
    response = requests.post(API_URL, json=test_case_2)
    if response.status_code == 200:
        result = response.json()
        prediction = result["prediction"]
        print(f"\n✅ API Response: {result}")
        if prediction == 0:
            print("🟢 Result: NORMAL - No heart disease detected")
        else:
            print("🔴 Result: HIGH RISK - Heart disease detected!")
    else:
        print(f"❌ Error: Status Code {response.status_code}")
except Exception as e:
    print(f"❌ Error: {e}")

# Test Case 3: Moderate Risk Patient
print("\n\n📋 Test Case 3: Moderate Risk Patient")
print("-" * 60)
test_case_3 = {
    "Age": 50,
    "Sex": "M",
    "ChestPainType": "TA",
    "RestingBP": 140,
    "Cholesterol": 220,
    "FastingBS": 0,
    "RestingECG": "ST",
    "MaxHR": 145,
    "ExerciseAngina": "N",
    "Oldpeak": 1.5,
    "ST_Slope": "Flat"
}

print("Input:")
for key, value in test_case_3.items():
    print(f"  {key}: {value}")

try:
    response = requests.post(API_URL, json=test_case_3)
    if response.status_code == 200:
        result = response.json()
        prediction = result["prediction"]
        print(f"\n✅ API Response: {result}")
        if prediction == 0:
            print("🟢 Result: NORMAL - No heart disease detected")
        else:
            print("🔴 Result: HIGH RISK - Heart disease detected!")
    else:
        print(f"❌ Error: Status Code {response.status_code}")
except Exception as e:
    print(f"❌ Error: {e}")

print("\n" + "=" * 60)
print("API Testing Complete!")
print("=" * 60)
print("\n🌐 Frontend URL: http://localhost:8501")
print("📚 API Documentation: http://127.0.0.1:8000/docs")
print("=" * 60)
