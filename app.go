package main

import (
	"os"
	"os/exec"
)

func main() {
	cmd := exec.Command("ffmpeg", "-i", "src/video.mp4", "dist/frame_%06d.png")
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	err := cmd.Run()
	if err != nil {
		panic(err)
	}
}
