"""
Initialize default data
Creates default users and regencies on first run
"""
from ..models import db, User, Regency
from flask import current_app


def _ensure_guest_user():
    """
    Ensure the default view-only 'guest' account exists.

    Runs on every startup (not just first init) so it also gets created in
    databases that were already initialized before the guest role existed.
    """
    if User.query.filter_by(username='guest').first() is not None:
        return

    guest = User(
        username='guest',
        email='guest@dengue-predict.local',
        role='guest'
    )
    guest.set_password('guest')
    db.session.add(guest)
    db.session.commit()
    print("Default guest account created (username: guest / password: guest)")


def initialize_default_data():
    """Initialize default data if database is empty"""

    _ensure_guest_user()

    # Check if other data already exists
    if User.query.filter(User.role != 'guest').first() is not None:
        return  # Data already initialized

    print("Initializing default data...")
    
    # Create default admin user
    admin = User(
        username='admin',
        email='admin@dengue-predict.local',
        role='admin'
    )
    admin.set_password('admin123')  # Change this in production!
    db.session.add(admin)
    
    # Create regencies from config
    regencies_config = current_app.config['REGENCIES']
    
    for regency_data in regencies_config:
        regency = Regency(
            name=regency_data['name'],
            latitude=regency_data['latitude'],
            longitude=regency_data['longitude'],
            population=regency_data.get('population'),
            area_km2=regency_data.get('area_km2'),
            is_active=True
        )
        db.session.add(regency)
    
    # Create sample health district users for each regency
    for regency_data in regencies_config:
        # Create username from regency name
        username = regency_data['name'].lower().replace(' ', '_').replace('kab_', '').replace('kota_', '')
        
        district_user = User(
            username=username,
            email=f"{username}@health.local",
            role='health_district',
            regency=regency_data['name']
        )
        district_user.set_password('health123')  # Change this in production!
        db.session.add(district_user)
    
    # Commit all changes
    try:
        db.session.commit()
        print("Default data initialized successfully!")
        print("\nDefault Users Created:")
        print("  Admin:")
        print("    Username: admin")
        print("    Password: admin123")
        print("  District Health Users:")
        for regency_data in regencies_config:
            username = regency_data['name'].lower().replace(' ', '_').replace('kab_', '').replace('kota_', '')
            print(f"    Username: {username}")
            print(f"    Password: health123")
            print(f"    Regency: {regency_data['name']}")
        print("  Guest:")
        print("    Username: guest")
        print("    Password: guest")
        print("\n⚠️  IMPORTANT: Change these default passwords before deploying to production!")
        
    except Exception as e:
        db.session.rollback()
        print(f"Error initializing default data: {str(e)}")
