import onnx
import onnxfbs.protos.TensorProto

def convert_tensor_proto(proto: onnx.TensorProto) -> onnxfbs.protos.TensorProto.TensorProtoT:
    buffer = onnxfbs.protos.TensorProto.TensorProtoT()
    buffer.dims = proto.dims
    buffer.dataType = proto.data_type
    buffer.segment = proto.segment
    buffer.floatData = proto.float_data
    buffer.int32Data = proto.int32_data
    buffer.stringData = proto.string_data
    buffer.int64Data = proto.int64_data
    buffer.name = proto.name
    buffer.docString = proto.doc_string
    buffer.rawData = proto.raw_data
    buffer.externalData = proto.external_data
    buffer.dataLocation = proto.data_location
    buffer.dataLocation = proto.data_location
    buffer.doubleData = proto.double_data
    buffer.uint64Data = proto.uint64_data
    return buffer
