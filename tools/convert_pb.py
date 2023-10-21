"""Convert onnx.proto to onnx.fbs

For flatc usage, see https://google.github.io/flatbuffers/flatbuffers_guide_using_schema_compiler.html
"""
import argparse
import os
import pathlib
import subprocess


def main(args):
    """Main function."""
    # Generate onnx.fbs
    subprocess.check_call(["flatc", args.proto, "--proto"])
    os.makedirs(args.outdir, exist_ok=True)
    path = pathlib.Path(args.outdir) / "onnx.fbs"
    os.rename("onnx.fbs", path)
    print("\n")
    print("🎉 Generated onnx.fbs at", path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--outdir", default=".", help="output directory")
    parser.add_argument("--proto", help="path to onnx.proto")
    args = parser.parse_args()
    main(args)
