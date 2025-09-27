"""
ONNX IR to FlatBuffers Serialization Examples

This module provides examples and usage documentation for the onnx_flatbuffers serialization module.
"""

# Examples module for onnx_flatbuffers serialization

def create_usage_example() -> str:
    """Create a comprehensive usage example showing how to serialize ONNX IR objects."""

    example_code = '''
"""
ONNX IR to FlatBuffers Serialization Examples
===========================================

This example demonstrates how to use the onnx_flatbuffers.serialization module
to convert ONNX IR objects to FlatBuffer format for efficient serialization.

The FlatBuffer format provides several advantages:
- Zero-copy deserialization
- Smaller memory footprint
- Faster loading times
- Cross-platform compatibility
- Forward/backward compatibility
"""

import onnx_ir
from onnx_flatbuffers.serialization import (
    serialize_model,
    serialize_graph,
    serialize_node,
    serialize_tensor,
    SerializationError
)

def basic_model_serialization():
    """Basic example of serializing an entire ONNX model."""
    try:
        # Load an ONNX model into IR format
        model = onnx_ir.load("path/to/your/model.onnx")

        # Serialize the model to FlatBuffer format
        flatbuffer_bytes = serialize_model(model)

        # Save the FlatBuffer data with .onnx2 extension
        with open("model.onnx2", "wb") as f:
            f.write(flatbuffer_bytes)

        print(f"Successfully serialized model to FlatBuffer format")
        print(f"  Original model: {model.producer_name} v{model.producer_version}")
        print(f"  IR Version: {model.ir_version}")
        print(f"  FlatBuffer size: {len(flatbuffer_bytes):,} bytes")

        return flatbuffer_bytes

    except FileNotFoundError:
        print("Error: Model file not found. Please provide a valid ONNX file path.")
        return None
    except SerializationError as e:
        print(f"Serialization failed: {e}")
        return None


def component_serialization():
    """Example of serializing individual ONNX components."""
    try:
        # Load a model
        model = onnx_ir.load("path/to/your/model.onnx")

        print("Serializing individual components...")

        # Serialize just the graph
        graph_bytes = serialize_graph(model.graph)
        with open("graph.fb", "wb") as f:
            f.write(graph_bytes)
        print(f"  Graph: {len(graph_bytes):,} bytes")

        # Serialize individual nodes
        node_count = 0
        for i, node in enumerate(model.graph):
            node_bytes = serialize_node(node)
            with open(f"node_{i}_{node.op_type}.fb", "wb") as f:
                f.write(node_bytes)
            node_count += 1

        print(f"  Nodes: {node_count} nodes serialized")

        # Serialize individual tensors (from initializers)
        tensor_count = 0
        if hasattr(model.graph, 'initializers'):
            for name, initializer in model.graph.initializers.items():
                if hasattr(initializer, 'const_value') and initializer.const_value:
                    tensor_bytes = serialize_tensor(initializer.const_value)
                    safe_name = name.replace("/", "_").replace(":", "_")
                    with open(f"tensor_{safe_name}.fb", "wb") as f:
                        f.write(tensor_bytes)
                    tensor_count += 1

        print(f"  Tensors: {tensor_count} tensors serialized")
        print("All components serialized successfully!")

    except Exception as e:
        print(f"Component serialization failed: {e}")


def batch_serialization():
    """Example of batch converting multiple ONNX models."""
    import glob
    import os

    # Find all ONNX files in a directory
    onnx_files = glob.glob("models/*.onnx")

    if not onnx_files:
        print("No ONNX files found in models/ directory")
        return

    successful_conversions = 0
    failed_conversions = 0

    for onnx_file in onnx_files:
        try:
            print(f"Processing {onnx_file}...")

            # Load model
            model = onnx_ir.load(onnx_file)

            # Serialize to FlatBuffer
            flatbuffer_bytes = serialize_model(model)

            # Create output filename
            base_name = os.path.splitext(onnx_file)[0]
            output_file = f"{base_name}.onnx2"

            # Save FlatBuffer data
            with open(output_file, "wb") as f:
                f.write(flatbuffer_bytes)

            print(f"  ✓ Converted {onnx_file} -> {output_file}")
            print(f"    Size: {len(flatbuffer_bytes):,} bytes")
            successful_conversions += 1

        except Exception as e:
            print(f"  ✗ Failed to convert {onnx_file}: {e}")
            failed_conversions += 1

    print(f"\\nBatch conversion complete:")
    print(f"  Successful: {successful_conversions}")
    print(f"  Failed: {failed_conversions}")


def performance_comparison():
    """Example showing performance benefits of FlatBuffer format."""
    import time
    import os

    model_path = "path/to/large_model.onnx"

    if not os.path.exists(model_path):
        print(f"Model file {model_path} not found for performance test")
        return

    try:
        print("Performance Comparison: Protobuf vs FlatBuffers")
        print("=" * 50)

        # Time loading original protobuf model
        start_time = time.time()
        model = onnx_ir.load(model_path)
        protobuf_load_time = time.time() - start_time

        # Get original file size
        original_size = os.path.getsize(model_path)

        # Time FlatBuffer serialization
        start_time = time.time()
        flatbuffer_bytes = serialize_model(model)
        serialization_time = time.time() - start_time

        # Save FlatBuffer for later loading test
        fb_path = model_path.replace(".onnx", ".onnx2")
        with open(fb_path, "wb") as f:
            f.write(flatbuffer_bytes)

        print(f"Original Protobuf:")
        print(f"  File size: {original_size:,} bytes")
        print(f"  Load time: {protobuf_load_time:.3f}s")

        print(f"\\nFlatBuffer:")
        print(f"  File size: {len(flatbuffer_bytes):,} bytes")
        print(f"  Serialization time: {serialization_time:.3f}s")

        size_ratio = len(flatbuffer_bytes) / original_size
        print(f"\\nSize comparison: {size_ratio:.2f}x (FlatBuffer/Protobuf)")

        if size_ratio < 1.0:
            print(f"FlatBuffer is {(1-size_ratio)*100:.1f}% smaller!")
        else:
            print(f"FlatBuffer is {(size_ratio-1)*100:.1f}% larger")

        print(f"\\nNote: FlatBuffer loading performance benefits are realized")
        print(f"during deserialization (not measured here)")

    except Exception as e:
        print(f"Performance test failed: {e}")


def inspect_serialized_model():
    """Example of inspecting a serialized FlatBuffer model."""
    try:
        # This would require implementing a deserializer
        print("Model inspection example:")
        print("Note: Full deserialization requires implementing FlatBuffer readers")
        print("For now, you can inspect the raw bytes:")

        with open("model.onnx2", "rb") as f:
            data = f.read()

        print(f"  File size: {len(data):,} bytes")
        print(f"  First 32 bytes (hex): {data[:32].hex()}")

        # Basic validation - FlatBuffer files start with a size prefix
        if len(data) >= 4:
            import struct
            size_prefix = struct.unpack('<I', data[:4])[0]
            print(f"  Size prefix: {size_prefix}")
            print(f"  Valid FlatBuffer: {size_prefix <= len(data) - 4}")

    except FileNotFoundError:
        print("No serialized model found. Run basic_model_serialization() first.")
    except Exception as e:
        print(f"Inspection failed: {e}")


def advanced_usage_patterns():
    """Advanced usage patterns and best practices."""

    print("Advanced Usage Patterns")
    print("=" * 30)

    print("\\n1. Custom Error Handling:")
    print("""
    try:
        model = onnx_ir.load("model.onnx")
        flatbuffer_bytes = serialize_model(model)
    except SerializationError as e:
        # Handle serialization-specific errors
        print(f"Serialization error: {e}")
    except Exception as e:
        # Handle other errors (file I/O, etc.)
        print(f"Unexpected error: {e}")
    """)

    print("\\n2. Memory-Efficient Streaming:")
    print("""
    # For very large models, consider processing in chunks
    def serialize_large_model(model):
        # Serialize components individually to manage memory
        components = []

        # Serialize graph
        graph_bytes = serialize_graph(model.graph)
        components.append(("graph", graph_bytes))

        # Serialize nodes individually
        for i, node in enumerate(model.graph):
            node_bytes = serialize_node(node)
            components.append((f"node_{i}", node_bytes))

        return components
    """)

    print("\\n3. Validation and Verification:")
    print("""
    def validate_serialization(original_model, flatbuffer_bytes):
        # Basic validation checks
        if len(flatbuffer_bytes) == 0:
            raise ValueError("Empty FlatBuffer output")

        # Size sanity check
        if len(flatbuffer_bytes) > 100 * 1024 * 1024:  # 100MB
            print("Warning: Very large FlatBuffer output")

        # TODO: Add semantic validation when deserializer is available
        return True
    """)

    print("\\n4. Integration with Model Pipelines:")
    print("""
    def convert_model_pipeline(input_path, output_path):
        # Complete pipeline for model conversion

        # Load and validate input
        model = onnx_ir.load(input_path)
        print(f"Loaded model: {model.graph.name or 'unnamed'}")

        # Serialize to FlatBuffer
        fb_bytes = serialize_model(model)

        # Save with metadata
        import json
        metadata = {
            "original_file": input_path,
            "conversion_time": time.time(),
            "ir_version": model.ir_version,
            "producer": model.producer_name,
            "flatbuffer_size": len(fb_bytes)
        }

        with open(output_path, "wb") as f:
            f.write(fb_bytes)

        with open(output_path + ".meta", "w") as f:
            json.dump(metadata, f, indent=2)

        return fb_bytes
    """)


if __name__ == "__main__":
    print("ONNX IR to FlatBuffers Serialization Examples")
    print("=" * 50)

    print("\\nAvailable examples:")
    print("1. basic_model_serialization() - Convert entire model")
    print("2. component_serialization() - Convert individual components")
    print("3. batch_serialization() - Convert multiple models")
    print("4. performance_comparison() - Compare with protobuf")
    print("5. inspect_serialized_model() - Examine FlatBuffer output")
    print("6. advanced_usage_patterns() - Best practices and patterns")

    print("\\nTo run examples, import this module and call the functions:")
    print("  from onnx_flatbuffers import examples")
    print("  examples.basic_model_serialization()")

    print("\\nFor more information, see the serialization module docstrings.")
'''

    return example_code


def print_api_reference():
    """Print API reference for the serialization module."""

    reference = '''
API Reference: onnx_flatbuffers.serialization
==========================================

Main Functions:
--------------

serialize_model(model: ModelProtocol) -> bytes
    Serialize an ONNX IR Model to FlatBuffer format.

    Args:
        model: ONNX IR Model object loaded with onnx_ir.load()

    Returns:
        bytes: Serialized FlatBuffer data

    Raises:
        SerializationError: If serialization fails

serialize_graph(graph: GraphProtocol) -> bytes
    Serialize an ONNX IR Graph to FlatBuffer format.

serialize_node(node: NodeProtocol) -> bytes
    Serialize an ONNX IR Node to FlatBuffer format.

serialize_tensor(tensor: TensorProtocol) -> bytes
    Serialize an ONNX IR Tensor to FlatBuffer format.

serialize_attribute(attribute: AttributeProtocol) -> bytes
    Serialize an ONNX IR Attribute to FlatBuffer format.

Exceptions:
----------

SerializationError(Exception)
    Raised when serialization fails due to invalid data or format issues.

Supported ONNX IR Features:
--------------------------

Models:
- IR version
- Producer name/version
- Domain and model version
- Opset imports
- Main graph
- Functions (partial support)
- Metadata properties
- Documentation strings

Graphs:
- Name and documentation
- Nodes (operators)
- Initializers (constant tensors)
- Inputs and outputs
- Value info

Nodes:
- Name and operator type
- Domain and overload
- Input/output names
- Attributes (all types)
- Documentation

Tensors:
- Name and data type
- Shape/dimensions
- Raw binary data
- Documentation

Attributes:
- All primitive types (int, float, string)
- All list types (ints, floats, strings)
- Tensor and graph attributes (future)

Data Types Supported:
-------------------
FLOAT, DOUBLE, INT8, INT16, INT32, INT64, UINT8, UINT16, UINT32, UINT64,
BOOL, STRING, FLOAT16, BFLOAT16, COMPLEX64, COMPLEX128, and newer types
like FLOAT8E4M3FN, UINT4, INT4, etc.

Usage Notes:
-----------
- FlatBuffers use zero-copy deserialization for better performance
- Files are forward/backward compatible within schema versions
- Memory layout is optimized for direct access without parsing
- Cross-platform binary format (little-endian)
- Suitable for deployment, caching, and high-performance inference

Performance Tips:
----------------
- Batch convert multiple models for efficiency
- Use component serialization for large models to manage memory
- FlatBuffer format is optimized for loading, not necessarily file size
- Consider compression (gzip, etc.) for storage if size is critical
'''

    return reference


if __name__ == "__main__":
    # Print examples when run directly
    print(create_usage_example())
    print("\\n" + "="*60 + "\\n")
    print(print_api_reference())