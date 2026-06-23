import asyncio
import logging
import os
import sys
import importlib.util
from importlib.machinery import SourceFileLoader
from unittest.mock import MagicMock

# Set up robust mocks using MagicMock
sys.modules['tensorflow'] = MagicMock()
sys.modules['tensorflow.keras'] = MagicMock()
sys.modules['tensorflow.keras.models'] = MagicMock()
sys.modules['tensorflow.keras.layers'] = MagicMock()
sys.modules['tensorflow.keras.metrics'] = MagicMock()
sys.modules['tensorflow.keras.callbacks'] = MagicMock()
sys.modules['tensorflow.keras.backend'] = MagicMock()
sys.modules['transformers'] = MagicMock()
sys.modules['nest_asyncio'] = MagicMock()
sys.modules['numpy'] = MagicMock()

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
    # Mock _verify_identity to return a string instead of a coroutine if it's called
    agi._verify_identity = MagicMock(return_value=asyncio.Future())
    agi._verify_identity.return_value.set_result("spiffe://test")

    # Mock _check_governance_policy
    agi._check_governance_policy = MagicMock(return_value=asyncio.Future())
    agi._check_governance_policy.return_value.set_result(True)

    # Mock _emit_audit_event
    agi._emit_audit_event = MagicMock(return_value=asyncio.Future())
    agi._emit_audit_event.return_value.set_result(None)

    await agi.load_model("test_model", "nlp_v1")

    if "nlp_v1" in agi.models:
        print("✓ Model loaded into memory.")
    else:
        print("✗ Model load failed.")

    # Check RCE nesting
    try:
        nested = agi.context_envelope.wrap()
        if nested.depth == 1:
            print("✓ RCE wrapping verified.")
        else:
            print(f"✗ RCE wrapping depth mismatch: {nested.depth}")
    except Exception as e:
        print(f"✗ RCE wrapping failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_cache_and_eaip())
