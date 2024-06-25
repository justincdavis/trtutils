# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
import os


def main():
    # get location of this script
    script_dir = os.path.dirname(os.path.realpath(__file__))

    # get location of the examples folder
    examples_dir = os.path.join(script_dir, "..", "examples")

    # fetch all the example files in the examples folder
    example_files = []
    for root, dirs, files in os.walk(examples_dir):
        for file in files:
            if file.startswith("_"):
                continue
            if file.endswith(".py"):
                example_files.append(os.path.join(root, file))

    os.makedirs(os.path.join("docs", "examples"), exist_ok=True)

    # create rst files for each example file in the same directory structure
    for example_file in example_files:
        # get the relative path of the example file
        relative_path = os.path.relpath(example_file, examples_dir)

        # create the path to the rst file
        rst_file = os.path.join("docs", "examples", relative_path.replace(".py", ".rst"))

        # create the directory for the rst file
        os.makedirs(os.path.dirname(rst_file), exist_ok=True)

        # create the rst file
        with open(rst_file, "w") as f:
            # f.write(f".. include:: {relative_path}\n")
            # f.write(f"    :literal:\n")
            # f.write(f"    :language: python\n")
            f.write(f".. _examples_{relative_path.replace('.py', '')}:\n\n")
            f.write(f"Example: {relative_path}\n")
            f.write(f"{'=' * (len(relative_path) + 9)}\n\n")
            f.write(f".. code-block:: python\n\n")
            with open(os.path.join("examples", relative_path)) as example_f:
                for line in example_f.readlines():
                    f.write(f"\t{line}")
            f.write("\n")

    # create the examples.rst file
    with open(os.path.join("docs", "examples.rst"), "w") as f:
        f.write("Examples\n")
        f.write("========\n\n")
        f.write(".. toctree::\n")
        f.write("    :maxdepth: 1\n\n")
        for example_file in example_files:
            # get the relative path of the example file
            relative_path = os.path.relpath(example_file, examples_dir)

            # create the path to the rst file
            rst_file = os.path.join("examples", relative_path.replace(".py", ".rst"))

            f.write(f"    {rst_file}\n")

if __name__ == "__main__":
    main()
