"""
Application Entry Point
Run this file to start the Flask application
"""
import os
from app import create_app

# Determine environment
env = os.environ.get('FLASK_ENV', 'development')

# Create app
app = create_app(env)

if __name__ == '__main__':
    # Get host and port from environment or use defaults
    host = os.environ.get('FLASK_HOST', '0.0.0.0')
    port = int(os.environ.get('FLASK_PORT', 5000))
    debug = env == 'development'
    
    print(f"""
    ╔════════════════════════════════════════════════════════╗
    ║     DENGUE PREDICTION WEB APPLICATION                  ║
    ║     AI-Powered Spatio-Temporal Prediction System       ║
    ╚════════════════════════════════════════════════════════╝
    
    Environment: {env.upper()}
    Running on: http://{host}:{port}
    
    Default Credentials:
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    Admin User:
      Username: admin
      Password: admin123
      
    District Health Users (one per regency):
      Username: bantul | gunung_kidul | kulon_progo | sleman | yogyakarta
      Password: health123
      
    ⚠️  IMPORTANT: Change these passwords before production deployment!
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    Press CTRL+C to stop the server
    """)
    
    app.run(host=host, port=port, debug=debug)
