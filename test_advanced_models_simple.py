"""
🧪 Simple Test for Advanced Models
Quick validation before running the full pipeline
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import sys
import os

# Add src to path
sys.path.append('src')

def test_basic_setup():
    """Test basic setup and imports"""
    print("🧪 Testing Advanced Models Setup...")
    
    try:
        # Test data loading
        df = pd.read_csv("data/raw/sample_creditcard.csv")
        print(f"✅ Data loaded: {df.shape}")
        
        # Prepare simple dataset
        X = df.drop('Class', axis=1)
        y = df['Class']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"✅ Data split completed:")
        print(f"   Training: {X_train.shape}")
        print(f"   Test: {X_test.shape}")
        print(f"   Fraud rate: {y.mean():.3%}")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic setup failed: {e}")
        return False

def test_imports():
    """Test if we can import our modules"""
    print("\n🔍 Testing Module Imports...")
    
    try:
        print("   Testing traditional models...")
        from models.traditional_models import TraditionalModels
        print("   ✅ Traditional models imported")
        
        print("   Testing data preprocessing...")
        from data.preprocessor import FraudDataPreprocessor
        print("   ✅ Data preprocessor imported")
        
        print("   All imports successful!")
        return True
        
    except ImportError as e:
        print(f"   ❌ Import failed: {e}")
        print("   Note: Some advanced imports may require additional packages")
        return False

def test_simple_training():
    """Test simple model training"""
    print("\n🏋️ Testing Simple Model Training...")
    
    try:
        # Load and prepare data
        df = pd.read_csv("data/raw/sample_creditcard.csv")
        X = df.drop('Class', axis=1)
        y = df['Class']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Test traditional models
        from models.traditional_models import TraditionalModels
        
        traditional = TraditionalModels()
        results = traditional.train_all_models(X_train, y_train)
        evaluation = traditional.evaluate_all_models(X_test, y_test)
        
        print("✅ Traditional models training successful!")
        
        for model_name, metrics in evaluation.items():
            print(f"   {model_name}: AUC = {metrics.get('auc_roc', 0):.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Simple training failed: {e}")
        return False

def main():
    """Run all basic tests"""
    print("🚀 FinGuard Advanced Models - Basic Testing")
    print("="*50)
    
    tests = [
        ("Basic Setup", test_basic_setup),
        ("Module Imports", test_imports),
        ("Simple Training", test_simple_training)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📝 Running: {test_name}")
        if test_func():
            passed += 1
        
    print(f"\n🏆 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Ready for advanced models!")
        print("\n🚀 Next steps:")
        print("   1. Install deep learning packages: pip install tensorflow torch ctgan")
        print("   2. Run: python scripts/train_advanced_models.py")
        print("   3. Watch your resume become irresistible!")
    else:
        print("⚠️ Some tests failed. Check the errors above.")

if __name__ == "__main__":
    main()
