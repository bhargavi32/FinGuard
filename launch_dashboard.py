"""
🚀 FinGuard Dashboard Launcher
Quick launcher for your portfolio dashboard
"""

import subprocess
import sys
import os

def check_requirements():
    """Check if required packages are installed"""
    required_packages = ['streamlit', 'plotly', 'requests']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} - OK")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package} - Missing")
    
    if missing_packages:
        print(f"\n⚠️ Missing packages: {', '.join(missing_packages)}")
        print("Installing missing packages...")
        
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install'] + missing_packages)
            print("✅ All packages installed successfully!")
        except subprocess.CalledProcessError:
            print("❌ Failed to install packages. Please install manually:")
            print(f"pip install {' '.join(missing_packages)}")
            return False
    
    return True

def launch_dashboard():
    """Launch the Streamlit dashboard"""
    print("🚀 Launching FinGuard Dashboard...")
    print("📊 This will open in your browser shortly...")
    print("🔗 URL: http://localhost:8501")
    print()
    print("💡 Tips:")
    print("   - Make sure your API server is running for full functionality")
    print("   - Use Ctrl+C to stop the dashboard")
    print()
    
    try:
        # Launch Streamlit
        subprocess.run([
            sys.executable, '-m', 'streamlit', 'run', 'dashboard_app.py',
            '--server.headless', 'false',
            '--server.address', 'localhost',
            '--server.port', '8501'
        ])
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped.")
    except Exception as e:
        print(f"❌ Error launching dashboard: {e}")

def main():
    print("🛡️ FinGuard Portfolio Dashboard Launcher")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not os.path.exists('dashboard_app.py'):
        print("❌ Error: dashboard_app.py not found!")
        print("Please run this script from the project root directory.")
        return
    
    # Check requirements
    print("\n📦 Checking requirements...")
    if not check_requirements():
        return
    
    print("\n🎯 Starting dashboard...")
    launch_dashboard()

if __name__ == "__main__":
    main()
