import subprocess
import pathlib
import cv2

# subprocess.run(["./split"])
# buffer
BUFFER = []

frames = pathlib.Path("frames")
frame_number = 1
for i in range(100):
    frame = cv2.imread(f"dist/frame_{frame_number:06d}.png")
    BUFFER.append(frame)
    frame_number += 1
print(BUFFER[0])
