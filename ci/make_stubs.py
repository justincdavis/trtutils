# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
import subprocess

def main():
    for module in ["pycuda", "tensorrt", "cuda"]:
        print(f"Making stubs for {module}")
        subprocess.run(["pyright", "--createstub", module])

    for module in ["cuda.cuda", "cuda.cudart"]:
        print(f"Making stubs for {module}")
        subprocess.run(["stubgen", "-o", "typings", "-m", module])

if __name__ == "__main__":
    main()
