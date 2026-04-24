import os
import pickle
import joblib

models_dir = "models"

print("=" * 60)
print("MODEL INSPECTION TOOL")
print("=" * 60)

# List all model files
model_files = [f for f in os.listdir(models_dir) if f.endswith('.pkl')]
print(f"\nFound {len(model_files)} model files:\n")

for filename in sorted(model_files):
    filepath = os.path.join(models_dir, filename)
    file_size = os.path.getsize(filepath)
    print(f"  • {filename:<50} ({file_size:,} bytes)")

print("\n" + "=" * 60)
print("LOADING AND INSPECTING MODELS")
print("=" * 60)

for filename in sorted(model_files):
    filepath = os.path.join(models_dir, filename)
    print(f"\n📦 {filename}")
    print("-" * 60)
    
    try:
        # Try loading with joblib first (preferred for sklearn models)
        try:
            model = joblib.load(filepath)
        except:
            # Fall back to pickle
            with open(filepath, 'rb') as f:
                model = pickle.load(f)
        
        print(f"  Type: {type(model).__name__}")
        print(f"  Module: {type(model).__module__}")
        
        # Show model-specific info
        if hasattr(model, 'get_params'):
            print(f"  ✓ Sklearn model detected")
            params = model.get_params()
            print(f"  Parameters: {len(params)} total")
        
        if hasattr(model, '__dict__'):
            print(f"  Attributes: {len(model.__dict__)}")
        
        print(f"  ✓ Successfully loaded")
        
    except Exception as e:
        print(f"  ✗ Error loading: {str(e)}")

print("\n" + "=" * 60)
print("HOW TO USE THESE MODELS IN YOUR CODE:")
print("=" * 60)
print("""
# Option 1: Load with joblib (recommended for sklearn models)
import joblib
model = joblib.load('models/ANN_Optimized.pkl')
predictions = model.predict(X_test)

# Option 2: Load with pickle
import pickle
with open('models/ANN_Optimized.pkl', 'rb') as f:
    model = pickle.load(f)

# The preprocessor is also available for data transformation
preprocessor = joblib.load('models/preprocessor.pkl')
X_transformed = preprocessor.transform(X)
""")
print("=" * 60)
