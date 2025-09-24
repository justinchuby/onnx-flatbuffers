"""
ONNX IR to FlatBuffers Serialization Package

This package provides functionality to serialize ONNX IR objects to FlatBuffer format
for efficient storage and transmission.

Main functions:
- serialize_model: Convert ONNX IR Model to FlatBuffer bytes
- serialize_graph: Convert ONNX IR Graph to FlatBuffer bytes
- serialize_node: Convert ONNX IR Node to FlatBuffer bytes
- serialize_tensor: Convert ONNX IR Tensor to FlatBuffer bytes
- serialize_attribute: Convert ONNX IR Attribute to FlatBuffer bytes

Example usage:
    ```python
    import onnx_ir
    from onnx_flatbuffers import serialize_model

    # Load ONNX model
    model = onnx_ir.load("model.onnx")

    # Serialize to FlatBuffer format
    flatbuffer_bytes = serialize_model(model)

    # Save as .onnx2 file
    with open("model.onnx2", "wb") as f:
        f.write(flatbuffer_bytes)
    ```
"""

from .serialization import (
    serialize_model,
    serialize_graph,
    serialize_node,
    serialize_tensor,
    serialize_attribute,
    SerializationError,
)

__all__ = [
    "serialize_model",
    "serialize_graph",
    "serialize_node",
    "serialize_tensor",
    "serialize_attribute",
    "SerializationError",
]

__version__ = "0.1.0"
