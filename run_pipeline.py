import subprocess

steps = [
    "python src/download_data.py",
    "python src/build_features.py",
    "python src/create_dataset.py",
    "python src/train.py",
    "python src/evaluate.py"
]

for step in steps:
    print("\nRunning:", step)
    subprocess.run(step, shell=True, check=True)

print("\nPipeline completed successfully.")