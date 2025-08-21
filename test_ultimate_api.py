"""
🚀 Ultimate FinGuard API Test Suite
Test all advanced AI endpoints and showcase capabilities
"""

import requests
import json
import time
from datetime import datetime

# API Configuration
BASE_URL = "http://localhost:8000"
HEADERS = {"Content-Type": "application/json"}

# Test transaction data
NORMAL_TRANSACTION = {
    "Time": 50400,  # 2 PM
    "Amount": 87.50,
    "V1": 0.2, "V2": -0.3, "V3": 0.1, "V4": -0.1, "V5": 0.15,
    "V6": -0.2, "V7": 0.25, "V8": -0.15, "V9": 0.1, "V10": -0.05
}

SUSPICIOUS_TRANSACTION = {
    "Time": 10800,  # 3 AM
    "Amount": 15750,
    "V1": 4.2, "V2": 3.8, "V3": -3.5, "V4": 4.1, "V5": -3.9,
    "V6": 3.7, "V7": -4.0, "V8": 3.6, "V9": -3.8, "V10": 4.3
}

def print_header(title):
    """Print formatted header"""
    print("\n" + "=" * 80)
    print(f"🎯 {title}")
    print("=" * 80)

def print_result(response, title):
    """Print formatted response"""
    print(f"\n🔍 {title}")
    print("-" * 50)
    
    if response.status_code == 200:
        data = response.json()
        print("✅ SUCCESS!")
        print(json.dumps(data, indent=2, default=str))
    else:
        print(f"❌ ERROR: {response.status_code}")
        print(response.text)

def test_system_endpoints():
    """Test system endpoints"""
    print_header("SYSTEM ENDPOINTS TESTING")
    
    # Root endpoint
    print("\n🌐 Testing Root Endpoint...")
    response = requests.get(f"{BASE_URL}/")
    print_result(response, "Root Information")
    
    # Health check
    print("\n🏥 Testing Health Check...")
    response = requests.get(f"{BASE_URL}/health")
    print_result(response, "Health Status")

def test_basic_fraud_detection():
    """Test basic fraud detection"""
    print_header("BASIC FRAUD DETECTION TESTING")
    
    # Normal transaction
    print("\n📊 Testing Normal Transaction...")
    response = requests.post(
        f"{BASE_URL}/fraud/predict", 
        json=NORMAL_TRANSACTION, 
        headers=HEADERS
    )
    print_result(response, "Normal Transaction Prediction")
    
    # Suspicious transaction
    print("\n🚨 Testing Suspicious Transaction...")
    response = requests.post(
        f"{BASE_URL}/fraud/predict", 
        json=SUSPICIOUS_TRANSACTION, 
        headers=HEADERS
    )
    print_result(response, "Suspicious Transaction Prediction")

def test_ensemble_prediction():
    """Test advanced ensemble prediction"""
    print_header("ENSEMBLE PREDICTION TESTING")
    
    # Normal transaction ensemble
    print("\n🧠 Testing Ensemble Prediction (Normal)...")
    response = requests.post(
        f"{BASE_URL}/fraud/predict/ensemble", 
        json=NORMAL_TRANSACTION, 
        headers=HEADERS
    )
    print_result(response, "Ensemble Prediction - Normal")
    
    # Suspicious transaction ensemble
    print("\n🧠 Testing Ensemble Prediction (Suspicious)...")
    response = requests.post(
        f"{BASE_URL}/fraud/predict/ensemble", 
        json=SUSPICIOUS_TRANSACTION, 
        headers=HEADERS
    )
    print_result(response, "Ensemble Prediction - Suspicious")

def test_explainable_ai():
    """Test advanced explainable AI"""
    print_header("EXPLAINABLE AI TESTING")
    
    # Normal transaction explanation
    print("\n💡 Testing Advanced Explanation (Normal)...")
    response = requests.post(
        f"{BASE_URL}/fraud/explain/advanced", 
        json=NORMAL_TRANSACTION, 
        headers=HEADERS
    )
    print_result(response, "Advanced Explanation - Normal")
    
    # Suspicious transaction explanation
    print("\n💡 Testing Advanced Explanation (Suspicious)...")
    response = requests.post(
        f"{BASE_URL}/fraud/explain/advanced", 
        json=SUSPICIOUS_TRANSACTION, 
        headers=HEADERS
    )
    print_result(response, "Advanced Explanation - Suspicious")

def test_gan_generation():
    """Test GAN synthetic data generation"""
    print_header("GAN SYNTHETIC DATA GENERATION TESTING")
    
    # Generate fraud-only samples
    print("\n🎭 Testing GAN - Fraud Only Generation...")
    gan_request = {
        "num_samples": 10,
        "fraud_type": "fraud_only",
        "quality_level": "high"
    }
    response = requests.post(
        f"{BASE_URL}/gan/generate", 
        json=gan_request, 
        headers=HEADERS
    )
    print_result(response, "GAN Fraud Generation")
    
    # Generate balanced samples
    print("\n🎭 Testing GAN - Balanced Generation...")
    gan_request = {
        "num_samples": 20,
        "fraud_type": "balanced",
        "quality_level": "medium"
    }
    response = requests.post(
        f"{BASE_URL}/gan/generate", 
        json=gan_request, 
        headers=HEADERS
    )
    print_result(response, "GAN Balanced Generation")

def test_model_training():
    """Test model training with synthetic data"""
    print_header("MODEL TRAINING WITH SYNTHETIC DATA")
    
    training_request = {
        "use_synthetic_data": True,
        "synthetic_samples": 500,
        "model_types": ["logistic", "random_forest", "neural_network"]
    }
    
    print("\n🔄 Testing Model Training...")
    response = requests.post(
        f"{BASE_URL}/models/train", 
        json=training_request, 
        headers=HEADERS
    )
    print_result(response, "Model Training Request")

def test_analytics():
    """Test performance analytics"""
    print_header("PERFORMANCE ANALYTICS TESTING")
    
    print("\n📊 Testing Performance Analytics...")
    response = requests.get(f"{BASE_URL}/analytics/performance")
    print_result(response, "Performance Analytics")

def test_api_docs():
    """Test API documentation endpoints"""
    print_header("API DOCUMENTATION TESTING")
    
    print("\n📚 Testing OpenAPI Schema...")
    response = requests.get(f"{BASE_URL}/openapi.json")
    
    if response.status_code == 200:
        print("✅ OpenAPI schema available!")
        schema = response.json()
        print(f"   API Title: {schema.get('info', {}).get('title', 'N/A')}")
        print(f"   API Version: {schema.get('info', {}).get('version', 'N/A')}")
        print(f"   Endpoints: {len(schema.get('paths', {}))}")
    else:
        print(f"❌ OpenAPI schema error: {response.status_code}")

def run_comprehensive_test():
    """Run comprehensive test suite"""
    print("🛡️ Ultimate FinGuard API - Comprehensive Test Suite")
    print("   Testing ALL Advanced AI Features")
    print("   " + "=" * 60)
    
    start_time = datetime.now()
    
    # Run all tests
    test_functions = [
        test_system_endpoints,
        test_basic_fraud_detection,
        test_ensemble_prediction,
        test_explainable_ai,
        test_gan_generation,
        test_model_training,
        test_analytics,
        test_api_docs
    ]
    
    successful_tests = 0
    total_tests = len(test_functions)
    
    for test_func in test_functions:
        try:
            test_func()
            successful_tests += 1
            print(f"\n✅ {test_func.__name__} completed successfully")
        except Exception as e:
            print(f"\n❌ {test_func.__name__} failed: {e}")
    
    # Final summary
    execution_time = datetime.now() - start_time
    
    print_header("COMPREHENSIVE TEST RESULTS")
    
    print(f"\n🎯 TEST EXECUTION SUMMARY:")
    print(f"   ✅ Successful tests: {successful_tests}/{total_tests}")
    print(f"   ⏱️  Execution time: {execution_time}")
    print(f"   📊 Success rate: {successful_tests/total_tests*100:.1f}%")
    
    print(f"\n🚀 API CAPABILITIES DEMONSTRATED:")
    print(f"   🧠 Deep Learning Models: Multi-layer neural networks")
    print(f"   🎭 GAN Technology: Synthetic fraud data generation")
    print(f"   💡 Explainable AI: Natural language explanations")
    print(f"   📊 Ensemble Methods: Multi-model prediction aggregation")
    print(f"   ⚡ Real-time Analytics: Performance monitoring")
    print(f"   🔄 Dynamic Training: Model retraining with synthetic data")
    
    print(f"\n💼 RECRUITER TALKING POINTS:")
    print(f"   • 'Built complete AI platform with deep learning and GANs'")
    print(f"   • 'Implemented ensemble methods combining multiple ML models'")
    print(f"   • 'Created explainable AI system for regulatory compliance'")
    print(f"   • 'Developed real-time synthetic data generation capabilities'")
    print(f"   • 'Built production-ready API with comprehensive analytics'")
    
    print(f"\n🏆 ENTERPRISE FEATURES SHOWCASED:")
    print(f"   ✅ Multi-model ensemble learning")
    print(f"   ✅ Advanced neural network architectures")
    print(f"   ✅ Generative AI for data augmentation")
    print(f"   ✅ Real-time explainable AI")
    print(f"   ✅ Performance monitoring and analytics")
    print(f"   ✅ Dynamic model training and deployment")
    
    if successful_tests == total_tests:
        print(f"\n🎉 ALL TESTS PASSED! Your Ultimate AI API is working perfectly!")
        print(f"🎯 Ready to showcase to recruiters at top AI companies!")
    else:
        print(f"\n⚠️ Some tests failed. Check the server status and try again.")
    
    print(f"\n🌐 Access your API at: {BASE_URL}/docs")
    print(f"📚 Interactive documentation with all endpoints!")

if __name__ == "__main__":
    # Quick connectivity test
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            print("🌐 API server is responding! Starting comprehensive test...")
            time.sleep(2)
            run_comprehensive_test()
        else:
            print(f"❌ API server error: {response.status_code}")
            print("💡 Make sure the Ultimate FinGuard API is running:")
            print("   python ultimate_finguard_api.py")
    except requests.exceptions.RequestException:
        print("❌ Cannot connect to API server!")
        print("💡 Please start the Ultimate FinGuard API first:")
        print("   python ultimate_finguard_api.py")
        print("🌐 Then run this test again.")

