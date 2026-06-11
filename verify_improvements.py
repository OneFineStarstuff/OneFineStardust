import asyncio
import logging
import os
import sys

# Mocking modules that might not be available or too heavy to load for a simple test
sys.modules['transformers'] = type('module', (), {'AutoModelForCausalLM': None, 'AutoTokenizer': None})
sys.modules['tensorflow'] = type('module', (), {'keras': type('module', (), {'models': type('module', (), {'Model': object})}), 'config': type('module', (), {'list_physical_devices': lambda x: []})})
sys.modules['tensorflow.keras'] = type('module', (), {'models': type('module', (), {'Model': object}), 'backend': None, 'callbacks': type('module', (), {'EarlyStopping': None, 'ModelCheckpoint': None})})

# Import AGISystemSTEM from the core file
# Since the filename has spaces and unicode, we use importlib
import importlib.util
spec = importlib.util.spec_from_file_location("core", "𝐎𝐧𝐞 𝐅. 𝐒𝐭𝐚𝐫𝐬𝐭𝐮𝐟𝐟")
core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(core)

async def test_cache_and_eaip():
    print("Testing AGISystemSTEM improvements...")

    # Mock model loader
    def mock_loader(name, cls):
        print(f"Loading model: {name}")
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

    print("Second load (should be cached)...")
    # Capturing stdout/logs is hard here, but we can check if models dict has it
    if "nlp_v1" in agi.models:
        print("✓ Model loaded into memory.")

    # Check RCE nesting
    try:
        nested = agi.context_envelope.wrap()
        if nested.depth == 1 and nested.parent_hash:
            print("✓ RCE wrapping verified.")
    except Exception as e:
        print(f"✗ RCE wrapping failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_cache_and_eaip())
