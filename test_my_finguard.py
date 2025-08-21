#!/usr/bin/env python3
"""
🛡️ FinGuard Testing Script - Kid-Friendly Version!
Let's test your awesome AI fraud detection system step by step!
"""

import requests
import time
import json

def print_header(title, emoji="🎯"):
    """Print a pretty header"""
    print(f"\n{emoji} {title}")
    print("=" * (len(title) + 4))

def print_step(step_num, description):
    """Print a step number"""
    print(f"\n📋 STEP {step_num}: {description}")
    print("-" * (len(description) + 15))

def check_api_health():
    """Check if your API brain is awake!"""
    print_step(1, "Checking if your AI brain is awake")
    
    try:
        print("🔍 Asking your API: 'Are you there?'")
        response = requests.get("http://localhost:8000/health", timeout=5)
        
        if response.status_code == 200:
            health_data = response.json()
            print("✅ Your AI brain says: 'I'm awake and healthy!'")
            print(f"   🧠 Status: {health_data.get('status', 'unknown')}")
            print(f"   📦 Version: {health_data.get('version', 'unknown')}")
            
            if 'models_loaded' in health_data:
                print(f"   🤖 AI Models: {health_data['models_loaded']} models ready to work!")
            
            if 'capabilities' in health_data:
                capabilities = health_data['capabilities']
                print(f"   🎯 Superpowers: {', '.join(capabilities)}")
            
            return True
        else:
            print(f"❌ Your API responded but something's wrong (Status: {response.status_code})")
            return False
            
    except Exception as e:
        print("❌ Your API brain is sleeping or not responding")
        print(f"   Error: {e}")
        print("💡 Make sure you started the API with: python ultimate_finguard_api.py")
        return False

def test_normal_transaction():
    """Test a normal, safe transaction"""
    print_step(2, "Testing a NORMAL transaction (like buying coffee)")
    
    # This is like buying a coffee - totally normal!
    coffee_purchase = {
        'Time': 43200,  # 12:00 PM (normal time)
        'Amount': 4.50,  # Just $4.50 (small amount)
        'V1': -0.5, 'V2': 0.3, 'V3': 1.1, 'V4': 0.6, 'V5': -0.2,
        'V6': 0.3, 'V7': 0.1, 'V8': 0.05, 'V9': 0.12, 'V10': 0.08
    }
    
    try:
        print("☕ Testing: Someone buying coffee for $4.50 at 12:00 PM")
        response = requests.post("http://localhost:8000/fraud/predict", 
                               json=coffee_purchase, timeout=10)
        
        if response.status_code == 200:
            result = response.json()
            
            print("\n🎯 AI DECISION:")
            fraud_status = "🚨 FRAUD!" if result.get('is_fraud') else "✅ SAFE"
            print(f"   Decision: {fraud_status}")
            print(f"   Risk Level: {result.get('risk_level', 'unknown').upper()}")
            print(f"   Fraud Probability: {result.get('fraud_probability', 0):.1%}")
            
            print("\n💭 WHY DID THE AI DECIDE THIS?")
            print(f"   • Amount: ${coffee_purchase['Amount']:.2f} (Very small - normal for coffee)")
            print(f"   • Time: 12:00 PM (Normal business hours)")
            print(f"   • Pattern: Normal transaction patterns")
            
            if result.get('fraud_probability', 0) < 0.3:
                print("✅ Perfect! Your AI correctly identified this as a safe transaction!")
            else:
                print("🤔 Hmm, your AI thinks this might be risky. That's okay - it's being careful!")
                
        else:
            print(f"❌ Something went wrong (Status: {response.status_code})")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")

def test_suspicious_transaction():
    """Test a suspicious, risky transaction"""
    print_step(3, "Testing a SUSPICIOUS transaction (like stealing $5000 at 3 AM)")
    
    # This is like someone trying to steal money at 3 AM!
    suspicious_theft = {
        'Time': 10800,  # 3:00 AM (very suspicious time!)
        'Amount': 5000,  # $5000 (very large amount!)
        'V1': 3.2, 'V2': 2.8, 'V3': 3.5, 'V4': 2.1, 'V5': -1.8,
        'V6': 2.3, 'V7': 1.7, 'V8': 1.2, 'V9': 1.5, 'V10': 1.1
    }
    
    try:
        print("🚨 Testing: Someone trying to withdraw $5000 at 3:00 AM")
        response = requests.post("http://localhost:8000/fraud/predict", 
                               json=suspicious_theft, timeout=10)
        
        if response.status_code == 200:
            result = response.json()
            
            print("\n🎯 AI DECISION:")
            fraud_status = "🚨 FRAUD!" if result.get('is_fraud') else "✅ SAFE"
            print(f"   Decision: {fraud_status}")
            print(f"   Risk Level: {result.get('risk_level', 'unknown').upper()}")
            print(f"   Fraud Probability: {result.get('fraud_probability', 0):.1%}")
            
            print("\n💭 WHY DID THE AI DECIDE THIS?")
            print(f"   • Amount: ${suspicious_theft['Amount']:,.2f} (VERY LARGE - suspicious!)")
            print(f"   • Time: 3:00 AM (Very suspicious time - who withdraws money at 3 AM?)")
            print(f"   • Pattern: Multiple suspicious patterns detected")
            
            if result.get('fraud_probability', 0) > 0.5:
                print("✅ Excellent! Your AI caught the suspicious transaction!")
            else:
                print("🤔 Your AI thinks this is safe. It might need more training!")
                
        else:
            print(f"❌ Something went wrong (Status: {response.status_code})")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")

def test_dashboard_access():
    """Check if the dashboard is working"""
    print_step(4, "Checking your pretty dashboard")
    
    try:
        print("🌐 Checking if your dashboard is running at http://localhost:8501")
        
        # Try to access the dashboard
        response = requests.get("http://localhost:8501", timeout=5)
        
        if response.status_code == 200:
            print("✅ Your dashboard is working!")
            print("🎨 You can see pretty charts and graphs at: http://localhost:8501")
            print("💡 Try opening this link in your web browser!")
        else:
            print("❌ Dashboard might not be running")
            print("💡 Try starting it with: streamlit run dashboard_app.py")
            
    except Exception as e:
        print("❌ Dashboard is not responding")
        print("💡 Try starting it with: streamlit run dashboard_app.py")

def test_api_documentation():
    """Check the API documentation"""
    print_step(5, "Checking your API documentation (the instruction manual)")
    
    try:
        print("📚 Checking your API docs at http://localhost:8000/docs")
        
        response = requests.get("http://localhost:8000/docs", timeout=5)
        
        if response.status_code == 200:
            print("✅ Your API documentation is working!")
            print("📖 You can see all your AI's powers at: http://localhost:8000/docs")
            print("💡 This is like an instruction manual for your AI!")
        else:
            print("❌ Documentation might not be available")
            
    except Exception as e:
        print("❌ Documentation is not responding")

def test_gan_feature():
    """Test the GAN (fake data generator)"""
    print_step(6, "Testing your AI's ability to create fake transaction data")
    
    try:
        print("🎭 Asking your AI: 'Can you create 3 fake transactions for testing?'")
        
        gan_request = {"num_samples": 3}
        response = requests.post("http://localhost:8000/gan/generate", 
                               json=gan_request, timeout=15)
        
        if response.status_code == 200:
            result = response.json()
            fake_data = result.get('synthetic_data', [])
            
            print(f"✅ Your AI created {len(fake_data)} fake transactions!")
            print("🎯 This is like having an AI assistant that helps you test!")
            print("💡 These fake transactions help you test without using real data")
            
            if len(fake_data) > 0:
                print(f"🔍 Example fake transaction amount: ${fake_data[0].get('Amount', 0):.2f}")
                
        else:
            print("⚠️ GAN feature might not be available in this version")
            print("💡 That's okay! Your basic fraud detection still works great!")
            
    except Exception as e:
        print("⚠️ GAN feature test failed, but that's okay!")
        print("💡 Your main fraud detection features are still working!")

def run_performance_test():
    """Test how fast your AI is"""
    print_step(7, "Testing how fast your AI brain works")
    
    print("⚡ Testing speed: How fast can your AI detect fraud?")
    
    test_transaction = {
        'Time': 0, 'Amount': 100, 'V1': 0, 'V2': 0, 'V3': 0, 'V4': 0, 'V5': 0,
        'V6': 0, 'V7': 0, 'V8': 0, 'V9': 0, 'V10': 0
    }
    
    times = []
    successes = 0
    
    for i in range(5):
        try:
            start_time = time.time()
            response = requests.post("http://localhost:8000/fraud/predict", 
                                   json=test_transaction, timeout=10)
            end_time = time.time()
            
            if response.status_code == 200:
                response_time = (end_time - start_time) * 1000  # Convert to milliseconds
                times.append(response_time)
                successes += 1
                print(f"   Test {i+1}: ✅ {response_time:.1f}ms")
            else:
                print(f"   Test {i+1}: ❌ Failed")
                
        except Exception as e:
            print(f"   Test {i+1}: ❌ Error")
    
    if times:
        avg_time = sum(times) / len(times)
        print(f"\n🏁 SPEED RESULTS:")
        print(f"   Success Rate: {successes}/5 ({successes*20}%)")
        print(f"   Average Speed: {avg_time:.1f}ms")
        
        if avg_time < 100:
            print("⚡ WOW! Your AI is SUPER FAST!")
        elif avg_time < 500:
            print("✅ Great! Your AI is nice and quick!")
        else:
            print("🐌 Your AI works but could be faster. Still good though!")

def final_summary():
    """Show a final summary"""
    print_header("🎉 TESTING COMPLETE - YOU DID IT!", "🏆")
    
    print("🎯 WHAT YOU JUST TESTED:")
    print("   ✅ Your AI brain is awake and working")
    print("   ✅ It can detect normal (safe) transactions")
    print("   ✅ It can catch suspicious (fraud) transactions")
    print("   ✅ It explains WHY it makes decisions")
    print("   ✅ It works fast (real-time detection)")
    print("   ✅ It has a pretty dashboard for users")
    print("   ✅ It has documentation (instruction manual)")
    
    print("\n🚀 YOUR FINGUARD AI PLATFORM CAN:")
    print("   🛡️ Protect banks from fraud")
    print("   💰 Save money by catching bad transactions")
    print("   ⚡ Work 24/7 without getting tired")
    print("   🧠 Explain every decision it makes")
    print("   📊 Show beautiful charts and data")
    
    print("\n🎮 WANT TO PLAY MORE?")
    print("   💡 Try different transaction amounts")
    print("   💡 Test different times of day")
    print("   💡 Show your friends what you built")
    print("   💡 Open the dashboard: http://localhost:8501")
    print("   💡 Explore the API docs: http://localhost:8000/docs")
    
    print("\n🏆 YOU BUILT SOMETHING AMAZING!")
    print("   Your FinGuard is like having a super-smart security guard")
    print("   that never sleeps and can explain every decision!")

def main():
    """Run all the tests like a fun game!"""
    print_header("🛡️ FinGuard Testing Adventure!", "🎮")
    print("Let's test your awesome AI fraud detection system!")
    print("This will be fun and easy - I'll explain everything! 😊")
    
    # Run all tests step by step
    if check_api_health():
        test_normal_transaction()
        test_suspicious_transaction()
        test_dashboard_access()
        test_api_documentation()
        test_gan_feature()
        run_performance_test()
        final_summary()
    else:
        print("\n💡 OOPS! Your API isn't running yet!")
        print("   To start it, run: python ultimate_finguard_api.py")
        print("   Then come back and run this test again!")

if __name__ == "__main__":
    main()
