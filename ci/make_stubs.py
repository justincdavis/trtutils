# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
import subprocess

def main():
    for module in ["pycuda", "tensorrt"]:
        print(f"Making stubs for {module}")
        subprocess.run(["pyright", "--createstub", module])

if __name__ == "__main__":
    main()
