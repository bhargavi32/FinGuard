#!/usr/bin/env python3
"""
Quick test of 2 key scenarios for FinGuard API
"""

import requests
import json

def test_two_cases():
    print("🛡️ FinGuard API - Testing 2 Key Cases")
    print("=" * 50)
    
    base_url = "http://localhost:8000"
    
    # Test Case 1: Normal Transaction
    print("\n🧪 Test Case 1: Normal Grocery Purchase ($127.50 at 8:00 AM)")
    case1 = {
        "Time": 28800,        # 8:00 AM (28800 seconds)
        "Amount": 127.50,     # Reasonable grocery amount
        "V1": -0.8, "V2": 0.6, "V3": 1.1, "V4": 0.3, "V5": -0.5,
        "V6": 0.4, "V7": 0.2, "V8": 0.1, "V9": 0.3, "V10": 0.2
    }
    
    try:
        response1 = requests.post(f"{base_url}/fraud/predict", json=case1, timeout=5)
        if response1.status_code == 200:
            result1 = response1.json()
            print(f"   🎯 Fraud Probability: {result1['fraud_probability']:.1%}")
            print(f"   📊 Risk Level: {result1['risk_level']}")
            print(f"   ⚖️  Decision: {'🚨 FRAUD' if result1['is_fraud'] else '✅ LEGITIMATE'}")
            print(f"   💬 Explanation: {result1['explanation']}")
            
            # Validate
            if result1['fraud_probability'] < 0.3:
                print("   ✅ CORRECT: Low fraud probability for normal transaction")
            else:
                print("   ⚠️  UNEXPECTED: High fraud probability for normal transaction")
        else:
            print(f"   ❌ Error: {response1.status_code}")
    except Exception as e:
        print(f"   ❌ Request failed: {e}")
    
    # Test Case 2: High-Risk Fraud
    print("\n🧪 Test Case 2: Suspicious Large Purchase ($4,850 at 4:00 AM)")
    case2 = {
        "Time": 14400,        # 4:00 AM (high-risk time)
        "Amount": 4850.00,    # Large amount
        "V1": 3.8, "V2": 4.2, "V3": 3.9, "V4": 3.5, "V5": -3.7,
        "V6": 3.6, "V7": 3.1, "V8": 2.8, "V9": 3.4, "V10": 3.2
    }
    
    try:
        response2 = requests.post(f"{base_url}/fraud/predict", json=case2, timeout=5)
        if response2.status_code == 200:
            result2 = response2.json()
            print(f"   🎯 Fraud Probability: {result2['fraud_probability']:.1%}")
            print(f"   📊 Risk Level: {result2['risk_level']}")
            print(f"   ⚖️  Decision: {'🚨 FRAUD' if result2['is_fraud'] else '✅ LEGITIMATE'}")
            print(f"   💬 Explanation: {result2['explanation']}")
            
            # Validate
            if result2['fraud_probability'] > 0.8:
                print("   ✅ CORRECT: High fraud probability for suspicious transaction")
            else:
                print("   ⚠️  UNEXPECTED: Low fraud probability for suspicious transaction")
        else:
            print(f"   ❌ Error: {response2.status_code}")
    except Exception as e:
        print(f"   ❌ Request failed: {e}")
    
    print(f"\n{'=' * 50}")
    print("🎯 **Test Summary:**")
    print("   Case 1: Normal transaction should have LOW fraud probability")
    print("   Case 2: Suspicious transaction should have HIGH fraud probability")
    print(f"\n🌐 Full API Documentation: {base_url}/docs")

if __name__ == "__main__":
    test_two_cases()
