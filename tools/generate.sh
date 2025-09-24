#!/usr/bin/env bash

# For flatc usage, see https://google.github.io/flatbuffers/flatbuffers_guide_using_schema_compiler.html
flatc -o onnx-flatbuffers/python/ $1 --gen-mutable --python --gen-onefile --python-typing
flatc -o onnx-flatbuffers/cpp $1 --gen-mutable --cpp --cpp-std c++17 --no-prefix --scoped-enums --natural-utf8 --gen-compare --gen-name-strings --reflect-types --reflect-names
flatc -o onnx-flatbuffers/rust $1 --gen-mutable --rust --reflect-types --reflect-names
