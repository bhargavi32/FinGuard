# 🛡️ FinGuard API Test Cases

## Test Case 1: Normal Legitimate Transaction
**Scenario**: Regular grocery store purchase during business hours

```json
{
  "Time": 28800,
  "Amount": 127.50,
  "V1": -0.8,
  "V2": 0.6,
  "V3": 1.1,
  "V4": 0.3,
  "V5": -0.5,
  "V6": 0.4,
  "V7": 0.2,
  "V8": 0.1,
  "V9": 0.3,
  "V10": 0.2
}
```

**Expected Result**: 
- Fraud Probability: < 20%
- Risk Level: MINIMAL or LOW
- Decision: LEGITIMATE
- Explanation: Normal transaction patterns

---

## Test Case 2: High-Risk Fraud Transaction
**Scenario**: Large unusual purchase at 4:00 AM with suspicious patterns

```json
{
  "Time": 14400,
  "Amount": 4850.00,
  "V1": 3.8,
  "V2": 4.2,
  "V3": 3.9,
  "V4": 3.5,
  "V5": -3.7,
  "V6": 3.6,
  "V7": 3.1,
  "V8": 2.8,
  "V9": 3.4,
  "V10": 3.2
}
```

**Expected Result**:
- Fraud Probability: > 85%
- Risk Level: CRITICAL
- Decision: FRAUD DETECTED
- Explanation: Multiple risk factors including high amount, suspicious timing, and anomalous patterns

---

## How to Test:

### Method 1: Using curl
```bash
curl -X POST "http://localhost:8000/fraud/predict" \
     -H "Content-Type: application/json" \
     -d '{"Time": 28800, "Amount": 127.50, "V1": -0.8, "V2": 0.6, "V3": 1.1, "V4": 0.3, "V5": -0.5, "V6": 0.4, "V7": 0.2, "V8": 0.1, "V9": 0.3, "V10": 0.2}'
```

### Method 2: Using Python
```python
import requests

test_data = {
    "Time": 28800,
    "Amount": 127.50,
    "V1": -0.8, "V2": 0.6, "V3": 1.1, "V4": 0.3, "V5": -0.5,
    "V6": 0.4, "V7": 0.2, "V8": 0.1, "V9": 0.3, "V10": 0.2
}

response = requests.post("http://localhost:8000/fraud/predict", json=test_data)
print(response.json())
```

### Method 3: Using API Documentation
Visit: http://localhost:8000/docs and use the interactive interface