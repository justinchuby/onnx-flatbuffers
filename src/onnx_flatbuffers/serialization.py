"""
ONNX IR to FlatBuffers Serialization Module

This module provides functions to convert ONNX IR objects (from the onnx_ir package)
to their corresponding FlatBuffer representations for efficient serialization.

The conversion follows the FlatBuffers tutorial at:
https://flatbuffers.dev/tutorial/#flatbufferbuilder

Example usage:
    ```python
    import onnx_ir
    from onnx_flatbuffers.serialization import serialize_model

    # Load an ONNX model into IR format
    model = onnx_ir.load("model.onnx")

    # Convert to FlatBuffer format
    flatbuffer_bytes = serialize_model(model)

    # Save the FlatBuffer data
    with open("model.onnx2", "wb") as f:
        f.write(flatbuffer_bytes)
    ```
"""

from __future__ import annotations

from typing import Any

import flatbuffers
from onnx_ir import _protocols

from . import fbs

__all__ = [
    "SerializationError",
    "serialize_model",
    "serialize_graph",
    "serialize_node",
    "serialize_tensor",
    "serialize_attribute",
]


class SerializationError(Exception):
    """Exception raised during ONNX IR to FlatBuffer serialization."""


def serialize_model(model: _protocols.ModelProtocol) -> bytearray:
    """Serialize an ONNX IR Model to FlatBuffer format.

    Args:
        model: The ONNX IR Model to serialize.

    Returns:
        The serialized FlatBuffer data as bytes.

    Raises:
        SerializationError: If serialization fails.
    """
    try:
        builder = flatbuffers.Builder(1024)

        # Convert the model to FlatBuffer offset
        model_offset = _convert_model(builder, model)

        # Finish the buffer with the model as root
        builder.Finish(model_offset)

        return builder.Output()

    except Exception as e:
        raise SerializationError(f"Failed to serialize model: {e}") from e


def serialize_graph(graph: _protocols.GraphProtocol) -> bytearray:
    """Serialize an ONNX IR Graph to FlatBuffer format.

    Args:
        graph: The ONNX IR Graph to serialize.

    Returns:
        The serialized FlatBuffer data as bytes.
    """
    try:
        builder = flatbuffers.Builder(1024)
        graph_offset = _convert_graph(builder, graph)
        builder.Finish(graph_offset)
        return builder.Output()
    except Exception as e:
        raise SerializationError(f"Failed to serialize graph: {e}") from e


def serialize_node(node: _protocols.NodeProtocol) -> bytearray:
    """Serialize an ONNX IR Node to FlatBuffer format.

    Args:
        node: The ONNX IR Node to serialize.

    Returns:
        The serialized FlatBuffer data as bytes.
    """
    try:
        builder = flatbuffers.Builder(1024)
        node_offset = _convert_node(builder, node)
        builder.Finish(node_offset)
        return builder.Output()
    except Exception as e:
        raise SerializationError(f"Failed to serialize node: {e}") from e


def serialize_tensor(tensor: _protocols.TensorProtocol) -> bytearray:
    """Serialize an ONNX IR Tensor to FlatBuffer format.

    Args:
        tensor: The ONNX IR Tensor to serialize.

    Returns:
        The serialized FlatBuffer data as bytes.
    """
    try:
        builder = flatbuffers.Builder(1024)
        tensor_offset = _convert_tensor(builder, tensor)
        builder.Finish(tensor_offset)
        return builder.Output()
    except Exception as e:
        raise SerializationError(f"Failed to serialize tensor: {e}") from e


def serialize_attribute(attribute: _protocols.AttributeProtocol) -> bytearray:
    """Serialize an ONNX IR Attribute to FlatBuffer format.

    Args:
        attribute: The ONNX IR Attribute to serialize.

    Returns:
        The serialized FlatBuffer data as bytes.
    """
    try:
        builder = flatbuffers.Builder(1024)
        attr_offset = _convert_attribute(builder, attribute)
        builder.Finish(attr_offset)
        return builder.Output()
    except Exception as e:
        raise SerializationError(f"Failed to serialize attribute: {e}") from e


def _convert_model(builder: flatbuffers.Builder, model: _protocols.ModelProtocol) -> int:
    """Convert an ONNX IR Model to FlatBuffer offset (internal helper).

    This follows the FlatBuffer pattern of:
    1. Create all child objects first (bottom-up)
    2. Create strings
    3. Create vectors
    4. Create the object itself
    """
    # Create strings first
    producer_name_offset = None
    if model.producer_name:
        producer_name_offset = builder.CreateString(model.producer_name)

    producer_version_offset = None
    if model.producer_version:
        producer_version_offset = builder.CreateString(model.producer_version)

    domain_offset = None
    if model.domain:
        domain_offset = builder.CreateString(model.domain)

    doc_string_offset = None
    if model.doc_string:
        doc_string_offset = builder.CreateString(model.doc_string)

    # Convert graph (required field)
    graph_offset = _convert_graph(builder, model.graph)

    # Convert opset imports vector
    opset_import_offsets = []
    if hasattr(model, 'opset_imports') and model.opset_imports:
        for opset in model.opset_imports:
            opset_offset = _convert_operator_set_id(builder, opset)
            opset_import_offsets.append(opset_offset)

    # Create opset import vector
    opset_import_vector = None
    if opset_import_offsets:
        fbs.ModelStartOpsetImportVector(builder, len(opset_import_offsets))
        for offset in reversed(opset_import_offsets):
            builder.PrependUOffsetTRelative(offset)
        opset_import_vector = builder.EndVector()

    # Convert functions vector
    functions_offsets = []
    if hasattr(model, 'functions') and model.functions:
        for func in model.functions:
            func_offset = _convert_function(builder, func)
            functions_offsets.append(func_offset)

    # Create functions vector
    functions_vector = None
    if functions_offsets:
        fbs.ModelStartFunctionsVector(builder, len(functions_offsets))
        for offset in reversed(functions_offsets):
            builder.PrependUOffsetTRelative(offset)
        functions_vector = builder.EndVector()

    # Convert metadata properties vector
    metadata_props_offsets = []
    if hasattr(model, 'metadata_props') and model.metadata_props:
        for prop in model.metadata_props:
            prop_offset = _convert_string_string_entry(builder, prop)
            metadata_props_offsets.append(prop_offset)

    # Create metadata props vector
    metadata_props_vector = None
    if metadata_props_offsets:
        fbs.ModelStartMetadataPropsVector(builder, len(metadata_props_offsets))
        for offset in reversed(metadata_props_offsets):
            builder.PrependUOffsetTRelative(offset)
        metadata_props_vector = builder.EndVector()

    # Create the Model
    fbs.ModelStart(builder)
    fbs.ModelAddIrVersion(builder, model.ir_version or fbs.Version.IR_VERSION)

    if opset_import_vector is not None:
        fbs.ModelAddOpsetImport(builder, opset_import_vector)

    if producer_name_offset is not None:
        fbs.ModelAddProducerName(builder, producer_name_offset)

    if producer_version_offset is not None:
        fbs.ModelAddProducerVersion(builder, producer_version_offset)

    if domain_offset is not None:
        fbs.ModelAddDomain(builder, domain_offset)

    if hasattr(model, 'model_version') and model.model_version:
        fbs.ModelAddModelVersion(builder, model.model_version)

    fbs.ModelAddGraph(builder, graph_offset)

    if functions_vector is not None:
        fbs.ModelAddFunctions(builder, functions_vector)

    if doc_string_offset is not None:
        fbs.ModelAddDocString(builder, doc_string_offset)

    if metadata_props_vector is not None:
        fbs.ModelAddMetadataProps(builder, metadata_props_vector)

    return fbs.ModelEnd(builder)


def _convert_graph(builder: flatbuffers.Builder, graph: _protocols.GraphProtocol) -> int:
    """Convert an ONNX IR Graph to FlatBuffer offset."""
    # Create strings
    name_offset = None
    if graph.name:
        name_offset = builder.CreateString(graph.name)

    doc_string_offset = None
    if graph.doc_string:
        doc_string_offset = builder.CreateString(graph.doc_string)

    # Convert nodes vector
    node_offsets = []
    for node in graph:  # Assuming graph is iterable over nodes
        node_offset = _convert_node(builder, node)
        node_offsets.append(node_offset)

    # Create nodes vector
    nodes_vector = None
    if node_offsets:
        fbs.GraphStartNodeVector(builder, len(node_offsets))
        for offset in reversed(node_offsets):
            builder.PrependUOffsetTRelative(offset)
        nodes_vector = builder.EndVector()

    # Convert initializers to tensors vector
    initializer_offsets = []
    if hasattr(graph, 'initializers'):
        for initializer in graph.initializers.values():
            if hasattr(initializer, 'const_value') and initializer.const_value:
                tensor_offset = _convert_tensor(builder, initializer.const_value)
                initializer_offsets.append(tensor_offset)

    # Create initializers vector
    initializers_vector = None
    if initializer_offsets:
        fbs.GraphStartInitializerVector(builder, len(initializer_offsets))
        for offset in reversed(initializer_offsets):
            builder.PrependUOffsetTRelative(offset)
        initializers_vector = builder.EndVector()

    # Convert inputs vector
    input_offsets = []
    if hasattr(graph, 'inputs'):
        for input_info in graph.inputs:
            input_offset = _convert_value_info(builder, input_info)
            input_offsets.append(input_offset)

    # Create inputs vector
    inputs_vector = None
    if input_offsets:
        fbs.GraphStartInputVector(builder, len(input_offsets))
        for offset in reversed(input_offsets):
            builder.PrependUOffsetTRelative(offset)
        inputs_vector = builder.EndVector()

    # Convert outputs vector
    output_offsets = []
    if hasattr(graph, 'outputs'):
        for output_info in graph.outputs:
            output_offset = _convert_value_info(builder, output_info)
            output_offsets.append(output_offset)

    # Create outputs vector
    outputs_vector = None
    if output_offsets:
        fbs.GraphStartOutputVector(builder, len(output_offsets))
        for offset in reversed(output_offsets):
            builder.PrependUOffsetTRelative(offset)
        outputs_vector = builder.EndVector()

    # Create the Graph
    fbs.GraphStart(builder)

    if name_offset is not None:
        fbs.GraphAddName(builder, name_offset)

    if nodes_vector is not None:
        fbs.GraphAddNode(builder, nodes_vector)

    if initializers_vector is not None:
        fbs.GraphAddInitializer(builder, initializers_vector)

    if inputs_vector is not None:
        fbs.GraphAddInput(builder, inputs_vector)

    if outputs_vector is not None:
        fbs.GraphAddOutput(builder, outputs_vector)

    if doc_string_offset is not None:
        fbs.GraphAddDocString(builder, doc_string_offset)

    return fbs.GraphEnd(builder)


def _convert_node(builder: flatbuffers.Builder, node: _protocols.NodeProtocol) -> int:
    """Convert an ONNX IR Node to FlatBuffer offset."""
    # Create strings
    name_offset = None
    if node.name:
        name_offset = builder.CreateString(node.name)

    op_type_offset = builder.CreateString(node.op_type)

    domain_offset = None
    if hasattr(node, 'domain') and node.domain:
        domain_offset = builder.CreateString(node.domain)

    doc_string_offset = None
    if hasattr(node, 'doc_string') and node.doc_string:
        doc_string_offset = builder.CreateString(node.doc_string)

    # Convert inputs vector
    input_offsets = []
    for input_name in node.inputs:
        if input_name:  # Skip empty input names
            input_offset = builder.CreateString(input_name)
            input_offsets.append(input_offset)

    # Create inputs vector
    inputs_vector = None
    if input_offsets:
        fbs.NodeStartInputVector(builder, len(input_offsets))
        for offset in reversed(input_offsets):
            builder.PrependUOffsetTRelative(offset)
        inputs_vector = builder.EndVector()

    # Convert outputs vector
    output_offsets = []
    for output_name in node.outputs:
        if output_name:  # Skip empty output names
            output_offset = builder.CreateString(output_name)
            output_offsets.append(output_offset)

    # Create outputs vector
    outputs_vector = None
    if output_offsets:
        fbs.NodeStartOutputVector(builder, len(output_offsets))
        for offset in reversed(output_offsets):
            builder.PrependUOffsetTRelative(offset)
        outputs_vector = builder.EndVector()

    # Convert attributes vector
    attribute_offsets = []
    if hasattr(node, 'attributes') and node.attributes:
        for attr_name, attr_value in node.attributes.items():
            # Create a simple attribute object from the name-value pair
            attr_offset = _convert_node_attribute(builder, attr_name, attr_value)
            attribute_offsets.append(attr_offset)

    # Create attributes vector
    attributes_vector = None
    if attribute_offsets:
        fbs.NodeStartAttributeVector(builder, len(attribute_offsets))
        for offset in reversed(attribute_offsets):
            builder.PrependUOffsetTRelative(offset)
        attributes_vector = builder.EndVector()

    # Create the Node
    fbs.NodeStart(builder)

    if name_offset is not None:
        fbs.NodeAddName(builder, name_offset)

    fbs.NodeAddOpType(builder, op_type_offset)

    if domain_offset is not None:
        fbs.NodeAddDomain(builder, domain_offset)

    if inputs_vector is not None:
        fbs.NodeAddInput(builder, inputs_vector)

    if outputs_vector is not None:
        fbs.NodeAddOutput(builder, outputs_vector)

    if attributes_vector is not None:
        fbs.NodeAddAttribute(builder, attributes_vector)

    if doc_string_offset is not None:
        fbs.NodeAddDocString(builder, doc_string_offset)

    return fbs.NodeEnd(builder)


def _convert_tensor(builder: flatbuffers.Builder, tensor: _protocols.TensorProtocol) -> int:
    """Convert an ONNX IR Tensor to FlatBuffer offset."""
    # Create strings
    name_offset = None
    if hasattr(tensor, 'name') and tensor.name:
        name_offset = builder.CreateString(tensor.name)

    doc_string_offset = None
    if hasattr(tensor, 'doc_string') and tensor.doc_string:
        doc_string_offset = builder.CreateString(tensor.doc_string)

    # Convert dims vector (use shape from tensor)
    dims_vector = None
    if hasattr(tensor, 'shape') and tensor.shape:
        # Convert shape to list of dimensions
        shape_dims = []
        for dim in tensor.shape:
            if isinstance(dim, int):
                shape_dims.append(dim)
            elif hasattr(dim, 'value') and isinstance(dim.value, int):
                shape_dims.append(dim.value)
            else:
                # For symbolic dimensions, use -1 as placeholder
                shape_dims.append(-1)

        if shape_dims:
            fbs.TensorStartDimsVector(builder, len(shape_dims))
            for dim in reversed(shape_dims):
                builder.PrependInt64(dim)
            dims_vector = builder.EndVector()

    # Convert raw data
    raw_data_vector = None
    if hasattr(tensor, 'raw') and tensor.raw:
        raw_data = tensor.raw
        fbs.TensorStartRawDataVector(builder, len(raw_data))
        for byte_val in reversed(raw_data):
            builder.PrependUint8(byte_val)
        raw_data_vector = builder.EndVector()

    # Map data type
    data_type = _map_data_type(getattr(tensor, 'data_type', None))

    # Create the Tensor
    fbs.TensorStart(builder)

    if name_offset is not None:
        fbs.TensorAddName(builder, name_offset)

    fbs.TensorAddDataType(builder, data_type)

    if dims_vector is not None:
        fbs.TensorAddDims(builder, dims_vector)

    if raw_data_vector is not None:
        fbs.TensorAddRawData(builder, raw_data_vector)

    if doc_string_offset is not None:
        fbs.TensorAddDocString(builder, doc_string_offset)

    return fbs.TensorEnd(builder)


def _convert_attribute(builder: flatbuffers.Builder, attribute: _protocols.AttributeProtocol) -> int:
    """Convert an ONNX IR Attribute to FlatBuffer offset."""
    # Create strings
    name_offset = builder.CreateString(attribute.name)

    doc_string_offset = None
    if hasattr(attribute, 'doc_string') and attribute.doc_string:
        doc_string_offset = builder.CreateString(attribute.doc_string)

    # Map attribute type and value
    attr_type = _map_attribute_type(attribute)

    # Create the Attribute
    fbs.AttributeStart(builder)
    fbs.AttributeAddName(builder, name_offset)
    fbs.AttributeAddType(builder, attr_type)

    # Add value based on type
    if hasattr(attribute, 'value'):
        if attr_type == fbs.AttributeType.FLOAT and isinstance(attribute.value, (int, float)):
            fbs.AttributeAddF(builder, float(attribute.value))
        elif attr_type == fbs.AttributeType.INT and isinstance(attribute.value, int):
            fbs.AttributeAddI(builder, attribute.value)
        elif attr_type == fbs.AttributeType.STRING and isinstance(attribute.value, str):
            string_offset = builder.CreateString(attribute.value)
            fbs.AttributeAddS(builder, string_offset)
        # TODO: Handle other attribute types (tensors, graphs, lists)

    if doc_string_offset is not None:
        fbs.AttributeAddDocString(builder, doc_string_offset)

    return fbs.AttributeEnd(builder)


# Helper functions for object conversions

def _convert_node_attribute(builder: flatbuffers.Builder, name: str, value: Any) -> int:
    """Convert a node attribute name-value pair to FlatBuffer offset."""
    # Create strings
    name_offset = builder.CreateString(name)

    # Determine attribute type and value based on the Python value
    if isinstance(value, str):
        attr_type = fbs.AttributeType.STRING
        value_offset = builder.CreateString(value)

        fbs.AttributeStart(builder)
        fbs.AttributeAddName(builder, name_offset)
        fbs.AttributeAddType(builder, attr_type)
        fbs.AttributeAddS(builder, value_offset)

    elif isinstance(value, int):
        attr_type = fbs.AttributeType.INT

        fbs.AttributeStart(builder)
        fbs.AttributeAddName(builder, name_offset)
        fbs.AttributeAddType(builder, attr_type)
        fbs.AttributeAddI(builder, value)

    elif isinstance(value, float):
        attr_type = fbs.AttributeType.FLOAT

        fbs.AttributeStart(builder)
        fbs.AttributeAddName(builder, name_offset)
        fbs.AttributeAddType(builder, attr_type)
        fbs.AttributeAddF(builder, value)

    elif isinstance(value, (list, tuple)):
        # Handle list values
        if value and isinstance(value[0], str):
            attr_type = fbs.AttributeType.STRINGS
            # Create string vector
            string_offsets = [builder.CreateString(s) for s in value]
            fbs.AttributeStartStringsVector(builder, len(string_offsets))
            for offset in reversed(string_offsets):
                builder.PrependUOffsetTRelative(offset)
            strings_vector = builder.EndVector()

            fbs.AttributeStart(builder)
            fbs.AttributeAddName(builder, name_offset)
            fbs.AttributeAddType(builder, attr_type)
            fbs.AttributeAddStrings(builder, strings_vector)

        elif value and isinstance(value[0], int):
            attr_type = fbs.AttributeType.INTS
            # Create int vector
            fbs.AttributeStartIntsVector(builder, len(value))
            for val in reversed(value):
                builder.PrependInt64(val)
            ints_vector = builder.EndVector()

            fbs.AttributeStart(builder)
            fbs.AttributeAddName(builder, name_offset)
            fbs.AttributeAddType(builder, attr_type)
            fbs.AttributeAddInts(builder, ints_vector)

        elif value and isinstance(value[0], float):
            attr_type = fbs.AttributeType.FLOATS
            # Create float vector
            fbs.AttributeStartFloatsVector(builder, len(value))
            for val in reversed(value):
                builder.PrependFloat32(val)
            floats_vector = builder.EndVector()

            fbs.AttributeStart(builder)
            fbs.AttributeAddName(builder, name_offset)
            fbs.AttributeAddType(builder, attr_type)
            fbs.AttributeAddFloats(builder, floats_vector)

        else:
            # Unknown list type, create as undefined
            attr_type = fbs.AttributeType.UNDEFINED
            fbs.AttributeStart(builder)
            fbs.AttributeAddName(builder, name_offset)
            fbs.AttributeAddType(builder, attr_type)

    else:
        # Unknown type, create as undefined
        attr_type = fbs.AttributeType.UNDEFINED
        fbs.AttributeStart(builder)
        fbs.AttributeAddName(builder, name_offset)
        fbs.AttributeAddType(builder, attr_type)

    return fbs.AttributeEnd(builder)


def _convert_operator_set_id(builder: flatbuffers.Builder, opset: Any) -> int:
    """Convert an OperatorSetId to FlatBuffer offset."""
    # Create strings
    domain_offset = None
    if hasattr(opset, 'domain') and opset.domain:
        domain_offset = builder.CreateString(opset.domain)

    # Create the OperatorSetId
    fbs.OperatorSetIdStart(builder)

    if domain_offset is not None:
        fbs.OperatorSetIdAddDomain(builder, domain_offset)

    if hasattr(opset, 'version'):
        fbs.OperatorSetIdAddVersion(builder, opset.version)

    return fbs.OperatorSetIdEnd(builder)


def _convert_function(builder: flatbuffers.Builder, function: Any) -> int:
    """Convert a Function to FlatBuffer offset."""
    # Placeholder implementation
    fbs.FunctionStart(builder)
    return fbs.FunctionEnd(builder)


def _convert_string_string_entry(builder: flatbuffers.Builder, entry: Any) -> int:
    """Convert a StringStringEntry to FlatBuffer offset."""
    # Create strings
    key_offset = None
    if hasattr(entry, 'key'):
        key_offset = builder.CreateString(entry.key)

    value_offset = None
    if hasattr(entry, 'value'):
        value_offset = builder.CreateString(entry.value)

    # Create the StringStringEntry
    fbs.StringStringEntryStart(builder)

    if key_offset is not None:
        fbs.StringStringEntryAddKey(builder, key_offset)

    if value_offset is not None:
        fbs.StringStringEntryAddValue(builder, value_offset)

    return fbs.StringStringEntryEnd(builder)


def _convert_value_info(builder: flatbuffers.Builder, value_info: Any) -> int:
    """Convert a ValueInfo to FlatBuffer offset."""
    # Placeholder implementation
    fbs.ValueInfoStart(builder)
    return fbs.ValueInfoEnd(builder)


# Type mapping functions

def _map_data_type(onnx_data_type: Any) -> int:
    """Map ONNX IR data type to FlatBuffer DataType enum."""
    if onnx_data_type is None:
        return fbs.DataType.UNDEFINED

    # If it's already a numeric value, return it directly (assuming it's valid)
    if isinstance(onnx_data_type, int):
        return onnx_data_type

    # Map common data types by name
    type_name = str(onnx_data_type).upper()

    # Complete mapping based on ONNX data types
    data_type_map = {
        'FLOAT': fbs.DataType.FLOAT,
        'FLOAT32': fbs.DataType.FLOAT,
        'UINT8': fbs.DataType.UINT8,
        'INT8': fbs.DataType.INT8,
        'UINT16': fbs.DataType.UINT16,
        'INT16': fbs.DataType.INT16,
        'INT32': fbs.DataType.INT32,
        'INT64': fbs.DataType.INT64,
        'STRING': fbs.DataType.STRING,
        'BOOL': fbs.DataType.BOOL,
        'BOOLEAN': fbs.DataType.BOOL,
        'FLOAT16': fbs.DataType.FLOAT16,
        'DOUBLE': fbs.DataType.DOUBLE,
        'FLOAT64': fbs.DataType.DOUBLE,
        'UINT32': fbs.DataType.UINT32,
        'UINT64': fbs.DataType.UINT64,
        'COMPLEX64': fbs.DataType.COMPLEX64,
        'COMPLEX128': fbs.DataType.COMPLEX128,
        'BFLOAT16': fbs.DataType.BFLOAT16,
        'FLOAT8E4M3FN': fbs.DataType.FLOAT8E4M3FN,
        'FLOAT8E4M3FNUZ': fbs.DataType.FLOAT8E4M3FNUZ,
        'FLOAT8E5M2': fbs.DataType.FLOAT8E5M2,
        'FLOAT8E5M2FNUZ': fbs.DataType.FLOAT8E5M2FNUZ,
        'UINT4': fbs.DataType.UINT4,
        'INT4': fbs.DataType.INT4,
        'FLOAT4E2M1': fbs.DataType.FLOAT4E2M1,
        'FLOAT8E8M0': fbs.DataType.FLOAT8E8M0,
    }

    return data_type_map.get(type_name, fbs.DataType.FLOAT)


def _map_attribute_type(attribute: _protocols.AttributeProtocol) -> int:
    """Map ONNX IR attribute type to FlatBuffer AttributeType enum."""
    if hasattr(attribute, 'type') and attribute.type:
        attr_type = str(attribute.type).upper()
        if 'FLOAT' in attr_type:
            return fbs.AttributeType.FLOAT
        elif 'INT' in attr_type:
            return fbs.AttributeType.INT
        elif 'STRING' in attr_type:
            return fbs.AttributeType.STRING
        elif 'TENSOR' in attr_type:
            return fbs.AttributeType.TENSOR
        elif 'GRAPH' in attr_type:
            return fbs.AttributeType.GRAPH

    # Infer type from value
    if hasattr(attribute, 'value') and attribute.value is not None:
        if isinstance(attribute.value, str):
            return fbs.AttributeType.STRING
        elif isinstance(attribute.value, (int, bool)):
            return fbs.AttributeType.INT
        elif isinstance(attribute.value, float):
            return fbs.AttributeType.FLOAT

    return fbs.AttributeType.UNDEFINED
