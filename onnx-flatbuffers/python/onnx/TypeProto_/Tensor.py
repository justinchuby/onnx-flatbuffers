# automatically generated by the FlatBuffers compiler, do not modify

# namespace: TypeProto_

import flatbuffers
from flatbuffers.compat import import_numpy
np = import_numpy()

class Tensor(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAs(cls, buf, offset=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = Tensor()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def GetRootAsTensor(cls, buf, offset=0):
        """This method is deprecated. Please switch to GetRootAs."""
        return cls.GetRootAs(buf, offset)
    # Tensor
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # Tensor
    def ElemType(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 0

    # Tensor
    def Shape(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            x = self._tab.Indirect(o + self._tab.Pos)
            from onnx.TensorShapeProto import TensorShapeProto
            obj = TensorShapeProto()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

def TensorStart(builder):
    builder.StartObject(2)

def Start(builder):
    TensorStart(builder)

def TensorAddElemType(builder, elemType):
    builder.PrependInt32Slot(0, elemType, 0)

def AddElemType(builder, elemType):
    TensorAddElemType(builder, elemType)

def TensorAddShape(builder, shape):
    builder.PrependUOffsetTRelativeSlot(1, flatbuffers.number_types.UOffsetTFlags.py_type(shape), 0)

def AddShape(builder, shape):
    TensorAddShape(builder, shape)

def TensorEnd(builder):
    return builder.EndObject()

def End(builder):
    return TensorEnd(builder)