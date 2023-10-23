# automatically generated by the FlatBuffers compiler, do not modify

# namespace: TypeProto_

import flatbuffers
from flatbuffers.compat import import_numpy
np = import_numpy()

class Optional(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAs(cls, buf, offset=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = Optional()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def GetRootAsOptional(cls, buf, offset=0):
        """This method is deprecated. Please switch to GetRootAs."""
        return cls.GetRootAs(buf, offset)
    # Optional
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # Optional
    def ElemType(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            x = self._tab.Indirect(o + self._tab.Pos)
            from onnxfbs.protos.TypeProto import TypeProto
            obj = TypeProto()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

def OptionalStart(builder):
    builder.StartObject(1)

def Start(builder):
    OptionalStart(builder)

def OptionalAddElemType(builder, elemType):
    builder.PrependUOffsetTRelativeSlot(0, flatbuffers.number_types.UOffsetTFlags.py_type(elemType), 0)

def AddElemType(builder, elemType):
    OptionalAddElemType(builder, elemType)

def OptionalEnd(builder):
    return builder.EndObject()

def End(builder):
    return OptionalEnd(builder)

import onnxfbs.protos.TypeProto
try:
    from typing import Optional
except:
    pass

class OptionalT(object):

    # OptionalT
    def __init__(self):
        self.elemType = None  # type: Optional[onnx.TypeProto.TypeProtoT]

    @classmethod
    def InitFromBuf(cls, buf, pos):
        optional = Optional()
        optional.Init(buf, pos)
        return cls.InitFromObj(optional)

    @classmethod
    def InitFromPackedBuf(cls, buf, pos=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, pos)
        return cls.InitFromBuf(buf, pos+n)

    @classmethod
    def InitFromObj(cls, optional):
        x = OptionalT()
        x._UnPack(optional)
        return x

    # OptionalT
    def _UnPack(self, optional):
        if optional is None:
            return
        if optional.ElemType() is not None:
            self.elemType = onnx.TypeProto.TypeProtoT.InitFromObj(optional.ElemType())

    # OptionalT
    def Pack(self, builder):
        if self.elemType is not None:
            elemType = self.elemType.Pack(builder)
        OptionalStart(builder)
        if self.elemType is not None:
            OptionalAddElemType(builder, elemType)
        optional = OptionalEnd(builder)
        return optional
