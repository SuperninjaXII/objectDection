import cv2
import numpy as np

# storage of a 100 images
BUFFER = []
for i in range(100):
    frame = cv2.imread(f"dist/frame_{i + 1:06d}.png", 0)
    if frame is not None:
        resized_frame = cv2.resize(frame, (28, 28))
        BUFFER.append(resized_frame)
# inputs
X = np.array(BUFFER).reshape(100, 1, 28, 28)
X = X / 255.0

# layer factory


class LayerConvolution:
    def __init__(self, n_filters, filter_size):
        self.n_filters = n_filters
        self.filter_size = filter_size
        self.weights = 0.01 * np.random.randn(n_filters, filter_size, filter_size)
        self.biases = np.zeros((n_filters, 1))

    def forward(self, inputs):
        self.inputs = inputs
        n_samples, channels, h, w = inputs.shape
        out_h = h - self.filter_size + 1
        out_w = w - self.filter_size + 1

        self.output = np.zeros((n_samples, self.n_filters, out_h, out_w))

        for i in range(out_h):
            for j in range(out_w):
                image_patch = inputs[
                    :, :, i : i + self.filter_size, j : j + self.filter_size
                ]

                for f in range(self.n_filters):
                    self.output[:, f, i, j] = (
                        np.sum(image_patch * self.weights[f], axis=(1, 2, 3))
                        + self.biases[f]
                    )


conv1 = LayerConvolution(n_filters=8, filter_size=3)

conv1.forward(X)

print(f"Original image shape: {X.shape}")
print(f"Convolution output shape: {conv1.output.shape}")
