from passlib.context import CryptContext
import sys

def test_hash():
    print("Initializing CryptContext...")
    try:
        pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        print("Hashing password...")
        hashed = pwd_context.hash("testpassword")
        print(f"Hash success: {hashed[:10]}...")
        
        print("Verifying password...")
        valid = pwd_context.verify("testpassword", hashed)
        print(f"Verify success: {valid}")
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_hash()
