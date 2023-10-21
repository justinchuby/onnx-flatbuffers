# automatically generated by the FlatBuffers compiler, do not modify

# namespace: onnx

import flatbuffers
from flatbuffers.compat import import_numpy
np = import_numpy()

class TensorShapeProto(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAs(cls, buf, offset=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = TensorShapeProto()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def GetRootAsTensorShapeProto(cls, buf, offset=0):
        """This method is deprecated. Please switch to GetRootAs."""
        return cls.GetRootAs(buf, offset)
    # TensorShapeProto
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # TensorShapeProto
    def Dim(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            x = self._tab.Vector(o)
            x += flatbuffers.number_types.UOffsetTFlags.py_type(j) * 4
            x = self._tab.Indirect(x)
            from onnx.TensorShapeProto_.Dimension import Dimension
            obj = Dimension()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

    # TensorShapeProto
    def DimLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # TensorShapeProto
    def DimIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        return o == 0

def TensorShapeProtoStart(builder):
    builder.StartObject(1)

def Start(builder):
    TensorShapeProtoStart(builder)

def TensorShapeProtoAddDim(builder, dim):
    builder.PrependUOffsetTRelativeSlot(0, flatbuffers.number_types.UOffsetTFlags.py_type(dim), 0)

def AddDim(builder, dim):
    TensorShapeProtoAddDim(builder, dim)

def TensorShapeProtoStartDimVector(builder, numElems):
    return builder.StartVector(4, numElems, 4)

def StartDimVector(builder, numElems: int) -> int:
    return TensorShapeProtoStartDimVector(builder, numElems)

def TensorShapeProtoEnd(builder):
    return builder.EndObject()

def End(builder):
    return TensorShapeProtoEnd(builder)
