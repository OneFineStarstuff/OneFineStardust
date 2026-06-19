import importlib.util
import os

filename = [f for f in os.listdir('.') if f.startswith('𝐎𝐧𝐞')][0]
print(f"Found core file: {filename}")
spec = importlib.util.spec_from_file_location("core", os.path.abspath(filename))
if spec is None:
    print("Spec is None")
else:
    core = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(core)
    print("Import successful")
    agi = core.AGISystemSTEM()
    print("AGISystemSTEM initialized")
