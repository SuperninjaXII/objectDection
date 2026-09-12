import subprocess
import pathlib
import cv2
import numpy

# subprocess.run(["./split"])
# buffer
BUFFER = []

frames = pathlib.Path("frames")
frame_number = 1

for i in range(100):
    frame = cv2.imread(f"dist/frame_{frame_number:06d}.png", 0)
    BUFFER.append(frame)
    frame_number += 1

# inputs
dataset = numpy.array(BUFFER[0])
m, n = dataset.shape
Xp = dataset[:, 0]
Yp = dataset[:, 1:]

print(Xp)
