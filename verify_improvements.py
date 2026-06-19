import asyncio
import logging
import os
import sys
import importlib.util
from importlib.machinery import SourceFileLoader

# Mocking modules that might not be available or too heavy to load for a simple test
sys.modules['transformers'] = type('module', (), {'AutoModelForCausalLM': None, 'AutoTokenizer': None})
sys.modules['tensorflow'] = type('module', (), {'keras': type('module', (), {'models': type('module', (), {'Model': object})}), 'config': type('module', (), {'list_physical_devices': lambda x: []}), 'errors': type('module', (), {'ResourceExhaustedError': Exception})})
sys.modules['tensorflow.keras'] = type('module', (), {'models': type('module', (), {'Model': object}), 'backend': None, 'callbacks': type('module', (), {'EarlyStopping': None, 'ModelCheckpoint': None})})
sys.modules['tensorflow.keras.models'] = type('module', (), {'Model': object})

# Find the core file
core_filename = [f for f in os.listdir('.') if f.startswith('𝐎𝐧𝐞')][0]
core_path = os.path.abspath(core_filename)

# Load the core module
loader = SourceFileLoader("core", core_path)
spec = importlib.util.spec_from_file_location("core", core_path, loader=loader)
core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(core)

async def test_cache_and_eaip():
    print(f"Testing AGISystemSTEM improvements using {core_filename}...")

    # Mock model loader
    def mock_loader(name, cls):
        return {"name": name}

    agi = core.AGISystemSTEM(model_loader=mock_loader)

    # Test RCE
    agi.update_context("test_agent", "initial state")
    if agi.context_envelope and len(agi.context_envelope.frames) == 1:
        print("✓ RCE update context verified.")
    else:
        print("✗ RCE update context failed.")

    # Test load_model cache and EAIP hooks
    print("First load...")
    await agi.load_model("test_model", "nlp_v1")

    if "nlp_v1" in agi.models:
        print("✓ Model loaded into memory.")
    else:
        print("✗ Model load failed.")

    # Check RCE nesting
    try:
        nested = agi.context_envelope.wrap()
        if nested.depth == 1 and nested.parent_hash:
            print("✓ RCE wrapping verified.")
    except Exception as e:
        print(f"✗ RCE wrapping failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_cache_and_eaip())
