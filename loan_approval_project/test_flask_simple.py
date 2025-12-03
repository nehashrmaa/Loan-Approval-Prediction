print("Testing Flask installation...")
try:
    from flask import Flask
    print("✅ Flask imported successfully!")
    
    app = Flask(__name__)
    
    @app.route('/')
    def hello():
        return "🎉 Flask is working correctly!"
    
    print("🚀 Starting test server...")
    print("🌐 Open: http://localhost:5000")
    app.run(debug=True, port=5000)
    
except ImportError as e:
    print(f"❌ Error: {e}")
    print("\nTry installing Flask with:")
    print("pip install --user flask")
except Exception as e:
    print(f"❌ Unexpected error: {e}")