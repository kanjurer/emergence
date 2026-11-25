import platform
import sys

import torch


def main():
    print("=== Emergence Environment ===")
    print("Python:", sys.version)
    print("Platform:", platform.platform())
    print("PyTorch:", torch.__version__)

    print("\n=== Hardware ===")
    print("MPS built:", torch.backends.mps.is_built())
    print("MPS available:", torch.backends.mps.is_available())

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    x = torch.randn(1000, 1000, device=device)
    y = x @ x

    print("Selected device:", device)
    print("Matrix multiplication:", y.shape)
    print("Environment check passed!")


if __name__ == "__main__":
    main()
