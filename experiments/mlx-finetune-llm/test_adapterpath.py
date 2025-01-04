from pathlib import Path
import json

# Get the directory of the current script
script_dir = Path(__file__).resolve().parent
print(script_dir)

# Construct the full path to the target file
adapter_path = script_dir / 'adapters'
print(adapter_path)

# Ensure the directory exists
# adapter_path.mkdir(parents=True, exist_ok=True)

adapter_file_path = adapter_path / 'adapter_config.json'
print(adapter_file_path)

#set LORA parameters
lora_config = {
 "lora_layers": 8,
 "num_layers": 8,
 "lora_parameters": {
    "rank": 8,
    "scale": 20.0,
    "dropout": 0.0,
}}

# Check if adapter_config.json exists
if adapter_file_path.exists():
    print(f"The file '{adapter_file_path}' already exists.")
else:
    with open(adapter_file_path, "x") as fid:
        json.dump(lora_config, fid, indent=4)
        print(f"Created '{adapter_file_path}'")
