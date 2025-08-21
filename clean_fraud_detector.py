#!/usr/bin/env python3
"""
Clean FinGuard Fraud Detection System
Follows the architecture diagram and gives accurate results
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import joblib
import os

class FinGuardFraudDetector:
    """Clean fraud detection system with proper sensitivity"""
    
    def __init__(self):
        self.models = {}
        self.scaler = None
        self.feature_names = None
        self.is_trained = False
        
    def load_and_prepare_data(self, data_path="data/raw/sample_creditcard.csv"):
        """Load and prepare fraud detection data"""
        print("📊 Loading fraud detection dataset...")
        
        # Load data
        if os.path.exists(data_path):
            df = pd.read_csv(data_path)
            print(f"✅ Loaded {len(df):,} transactions")
            print(f"📈 Fraud rate: {df['Class'].mean()*100:.2f}%")
        else:
            # Create synthetic realistic data if file doesn't exist
            print("⚠️ Sample data not found, creating synthetic data...")
            df = self._create_realistic_fraud_data()
        
        return df
    
    def _create_realistic_fraud_data(self):
        """Create realistic fraud detection data"""
        np.random.seed(42)
        n_samples = 10000
        
        # Create realistic fraud patterns
        data = []
        
        for i in range(n_samples):
            # 5% fraud rate (realistic)
            is_fraud = np.random.random() < 0.05
            
            if is_fraud:
                # Fraudulent transaction patterns
                amount = np.random.lognormal(8, 1.5)  # Higher amounts
                time = np.random.choice([
                    np.random.uniform(0, 21600),      # Late night
                    np.random.uniform(82800, 86400)   # Very late
                ])
                # Anomalous V features
                v_features = np.random.normal(0, 3, 10)  # More extreme values
            else:
                # Normal transaction patterns  
                amount = np.random.lognormal(4, 1)  # Normal amounts
                time = np.random.uniform(21600, 82800)  # Business hours
                # Normal V features
                v_features = np.random.normal(0, 1, 10)  # Normal values
            
            row = {
                'Time': time,
                'Amount': amount,
                **{f'V{i+1}': v_features[i] for i in range(10)},
                'Class': int(is_fraud)
            }
            data.append(row)
        
        df = pd.DataFrame(data)
        print(f"✅ Created {len(df):,} synthetic transactions")
        print(f"📈 Fraud rate: {df['Class'].mean()*100:.2f}%")
        return df
    
    def train_models(self, df):
        """Train fraud detection models with proper sensitivity"""
        print("\n🧠 Training fraud detection models...")
        
        # Prepare features
        X = df.drop('Class', axis=1)
        y = df['Class']
        self.feature_names = X.columns.tolist()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train models with proper fraud sensitivity
        models_config = {
            'logistic_regression': LogisticRegression(
                random_state=42, 
                class_weight='balanced'  # Handle imbalanced data
            ),
            'random_forest': RandomForestClassifier(
                n_estimators=100, 
                random_state=42,
                class_weight='balanced',  # Handle imbalanced data
                max_depth=10
            )
        }
        
        # Train and evaluate models
        for name, model in models_config.items():
            print(f"   Training {name}...")
            model.fit(X_train_scaled, y_train)
            
            # Evaluate
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            auc = roc_auc_score(y_test, y_pred_proba)
            print(f"     AUC: {auc:.4f}")
            
            self.models[name] = model
        
        self.is_trained = True
        print("✅ Models trained successfully!")
        
        # Save models
        self._save_models()
        
        return X_test_scaled, y_test
    
    def predict_fraud(self, transaction_data):
        """Predict fraud with proper sensitivity and explanations"""
        if not self.is_trained:
            raise ValueError("Models not trained yet!")
        
        # Prepare transaction data
        if isinstance(transaction_data, dict):
            # Convert dict to DataFrame
            df = pd.DataFrame([transaction_data])
        else:
            df = transaction_data.copy()
        
        # Ensure all required features are present
        for feature in self.feature_names:
            if feature not in df.columns:
                df[feature] = 0
        
        # Scale features
        X_scaled = self.scaler.transform(df[self.feature_names])
        
        # Ensemble prediction (weighted average)
        predictions = {}
        probabilities = []
        
        for name, model in self.models.items():
            prob = model.predict_proba(X_scaled)[0, 1]
            predictions[name] = {
                'fraud_probability': float(prob),
                'is_fraud': bool(prob > 0.5)
            }
            probabilities.append(prob)
        
        # Ensemble result (average of models)
        ensemble_prob = np.mean(probabilities)
        
        # Enhanced rule-based adjustment for obvious fraud patterns
        rule_bonus = self._calculate_rule_bonus(transaction_data)
        final_prob = min(ensemble_prob + rule_bonus, 0.99)
        
        # Determine risk level
        if final_prob >= 0.8:
            risk_level = "critical"
        elif final_prob >= 0.6:
            risk_level = "high"  
        elif final_prob >= 0.4:
            risk_level = "medium"
        elif final_prob >= 0.2:
            risk_level = "low"
        else:
            risk_level = "minimal"
        
        # Generate explanation
        explanation = self._generate_explanation(transaction_data, final_prob)
        
        return {
            'is_fraud': bool(final_prob > 0.5),
            'fraud_probability': float(final_prob),
            'risk_level': risk_level,
            'confidence_score': float(abs(final_prob - 0.5) * 2),
            'explanation': explanation,
            'model_predictions': predictions,
            'ensemble_probability': float(ensemble_prob),
            'rule_adjustment': float(rule_bonus)
        }
    
    def _calculate_rule_bonus(self, transaction):
        """Calculate rule-based fraud score bonus for obvious patterns"""
        bonus = 0.0
        
        amount = float(transaction.get('Amount', 0))
        time_val = float(transaction.get('Time', 0))
        
        # Amount-based risk
        if amount > 5000:
            bonus += 0.3
        elif amount > 3000:
            bonus += 0.2
        elif amount > 2000:
            bonus += 0.1
        
        # Time-based risk  
        hour = (time_val % 86400) // 3600
        if hour < 6 or hour > 22:
            bonus += 0.25
        elif hour < 8 or hour > 20:
            bonus += 0.1
        
        # Pattern-based risk (V features)
        high_v_count = 0
        for i in range(1, 11):
            v_key = f'V{i}'
            if v_key in transaction:
                v_val = abs(float(transaction[v_key]))
                if v_val > 3:
                    high_v_count += 1
                    bonus += 0.05
                elif v_val > 2:
                    high_v_count += 1
                    bonus += 0.02
        
        # Bonus for multiple anomalous features
        if high_v_count >= 5:
            bonus += 0.2
        elif high_v_count >= 3:
            bonus += 0.1
        
        return bonus
    
    def _generate_explanation(self, transaction, fraud_prob):
        """Generate human-readable explanation"""
        amount = float(transaction.get('Amount', 0))
        time_val = float(transaction.get('Time', 0))
        hour = int((time_val % 86400) // 3600)
        
        explanations = []
        
        # Amount explanation
        if amount > 3000:
            explanations.append(f"Very high transaction amount: ${amount:,.2f}")
        elif amount > 1000:
            explanations.append(f"Large transaction amount: ${amount:,.2f}")
        
        # Time explanation
        if hour < 6 or hour > 22:
            explanations.append(f"Suspicious timing: {hour:02d}:00 (outside normal hours)")
        
        # Pattern explanation
        high_v_features = sum(1 for i in range(1, 11) 
                             if abs(float(transaction.get(f'V{i}', 0))) > 2)
        if high_v_features >= 3:
            explanations.append(f"Unusual transaction patterns detected ({high_v_features} anomalous features)")
        
        if fraud_prob > 0.7:
            base_msg = "HIGH FRAUD RISK detected"
        elif fraud_prob > 0.4:
            base_msg = "MODERATE fraud risk detected"
        elif fraud_prob > 0.2:
            base_msg = "LOW fraud risk detected"
        else:
            base_msg = "Transaction appears legitimate"
        
        if explanations:
            return f"{base_msg} due to: {'; '.join(explanations)}"
        else:
            return f"{base_msg} - all indicators within normal ranges"
    
    def _save_models(self):
        """Save trained models"""
        os.makedirs('models', exist_ok=True)
        
        # Save models and scaler
        for name, model in self.models.items():
            joblib.dump(model, f'models/{name}.joblib')
        
        joblib.dump(self.scaler, 'models/scaler.joblib')
        joblib.dump(self.feature_names, 'models/feature_names.joblib')
        
        print("💾 Models saved to models/ directory")
    
    def load_models(self):
        """Load pre-trained models"""
        try:
            for name in ['logistic_regression', 'random_forest']:
                self.models[name] = joblib.load(f'models/{name}.joblib')
            
            self.scaler = joblib.load('models/scaler.joblib')
            self.feature_names = joblib.load('models/feature_names.joblib')
            self.is_trained = True
            print("✅ Models loaded successfully!")
            return True
        except FileNotFoundError:
            print("⚠️ No saved models found. Need to train first.")
            return False

def main():
    """Train and test the fraud detection system"""
    print("🛡️ FinGuard Clean Fraud Detection System")
    print("=" * 50)
    
    # Initialize detector
    detector = FinGuardFraudDetector()
    
    # Try to load existing models
    if not detector.load_models():
        # Train new models
        df = detector.load_and_prepare_data()
        detector.train_models(df)
    
    # Test with the problematic cases
    print("\n🧪 Testing fraud detection accuracy...")
    
    test_cases = [
        {
            "name": "Normal Coffee Purchase",
            "data": {
                "Time": 43200, "Amount": 4.75,
                "V1": -0.5, "V2": 0.3, "V3": 0.8, "V4": 0.4, "V5": -0.2,
                "V6": 0.1, "V7": 0.2, "V8": 0.05, "V9": 0.1, "V10": 0.08
            },
            "expected": "LOW"
        },
        {
            "name": "Suspicious Late Night",
            "data": {
                "Time": 9000, "Amount": 2800,
                "V1": 2.8, "V2": 3.1, "V3": 2.9, "V4": 2.4, "V5": -2.1,
                "V6": 2.6, "V7": 1.9, "V8": 1.7, "V9": 2.2, "V10": 1.8
            },
            "expected": "HIGH"
        },
        {
            "name": "Extreme Fraud Attempt",
            "data": {
                "Time": 13500, "Amount": 7500,
                "V1": 4.2, "V2": 3.8, "V3": 4.5, "V4": 3.6, "V5": -3.2,
                "V6": 4.1, "V7": 3.4, "V8": 2.9, "V9": 3.7, "V10": 3.1
            },
            "expected": "CRITICAL"
        }
    ]
    
    for test in test_cases:
        result = detector.predict_fraud(test["data"])
        print(f"\n🔍 {test['name']}:")
        print(f"   Fraud Probability: {result['fraud_probability']:.1%}")
        print(f"   Risk Level: {result['risk_level'].upper()}")
        print(f"   Expected: {test['expected']}")
        print(f"   Decision: {'🚨 FRAUD' if result['is_fraud'] else '✅ SAFE'}")
        print(f"   Explanation: {result['explanation']}")
        
        # Check if result matches expectation
        if (test['expected'] == 'LOW' and result['fraud_probability'] < 0.3) or \
           (test['expected'] == 'HIGH' and result['fraud_probability'] > 0.6) or \
           (test['expected'] == 'CRITICAL' and result['fraud_probability'] > 0.8):
            print("   ✅ CORRECT DETECTION!")
        else:
            print("   ⚠️ Needs adjustment")
    
    print(f"\n🎯 Fraud detection system ready!")
    print(f"💾 Models saved for API use")

if __name__ == "__main__":
    main()
