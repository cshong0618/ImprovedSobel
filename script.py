import numpy as np
import os
import sys

os.system("which python")

if len(sys.argv) == 1:
    print("No filename provided")
    exit()

filename = sys.argv[1]
filename_without_extention = filename.rsplit(".")[0]

for i in np.arange(1, 16):
    command = f"python main.py --input={filename} --output={filename_without_extention}_753_{i}.png --removal_scale={i}"
    print(f"Running >> {command}")
    os.system(command)