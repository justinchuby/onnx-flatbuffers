# automatically generated by the FlatBuffers compiler, do not modify

# namespace: TensorShapeProto_

import flatbuffers
from flatbuffers.compat import import_numpy
np = import_numpy()

class Dimension(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAs(cls, buf, offset=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = Dimension()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def GetRootAsDimension(cls, buf, offset=0):
        """This method is deprecated. Please switch to GetRootAs."""
        return cls.GetRootAs(buf, offset)
    # Dimension
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # Dimension
    def Value(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            x = self._tab.Indirect(o + self._tab.Pos)
            from onnx.TensorShapeProto_.Dimension_.Anonymous0 import Anonymous0
            obj = Anonymous0()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

    # Dimension
    def Denotation(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.String(o + self._tab.Pos)
        return None

def DimensionStart(builder):
    builder.StartObject(2)

def Start(builder):
    DimensionStart(builder)

def DimensionAddValue(builder, value):
    builder.PrependUOffsetTRelativeSlot(0, flatbuffers.number_types.UOffsetTFlags.py_type(value), 0)

def AddValue(builder, value):
    DimensionAddValue(builder, value)

def DimensionAddDenotation(builder, denotation):
    builder.PrependUOffsetTRelativeSlot(1, flatbuffers.number_types.UOffsetTFlags.py_type(denotation), 0)

def AddDenotation(builder, denotation):
    DimensionAddDenotation(builder, denotation)

def DimensionEnd(builder):
    return builder.EndObject()

def End(builder):
    return DimensionEnd(builder)

import onnx.TensorShapeProto_.Dimension_.Anonymous0
try:
    from typing import Optional
except:
    pass

class DimensionT(object):

    # DimensionT
    def __init__(self):
        self.value = None  # type: Optional[onnx.TensorShapeProto_.Dimension_.Anonymous0.Anonymous0T]
        self.denotation = None  # type: str

    @classmethod
    def InitFromBuf(cls, buf, pos):
        dimension = Dimension()
        dimension.Init(buf, pos)
        return cls.InitFromObj(dimension)

    @classmethod
    def InitFromPackedBuf(cls, buf, pos=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, pos)
        return cls.InitFromBuf(buf, pos+n)

    @classmethod
    def InitFromObj(cls, dimension):
        x = DimensionT()
        x._UnPack(dimension)
        return x

    # DimensionT
    def _UnPack(self, dimension):
        if dimension is None:
            return
        if dimension.Value() is not None:
            self.value = onnx.TensorShapeProto_.Dimension_.Anonymous0.Anonymous0T.InitFromObj(dimension.Value())
        self.denotation = dimension.Denotation()

    # DimensionT
    def Pack(self, builder):
        if self.value is not None:
            value = self.value.Pack(builder)
        if self.denotation is not None:
            denotation = builder.CreateString(self.denotation)
        DimensionStart(builder)
        if self.value is not None:
            DimensionAddValue(builder, value)
        if self.denotation is not None:
            DimensionAddDenotation(builder, denotation)
        dimension = DimensionEnd(builder)
        return dimension
