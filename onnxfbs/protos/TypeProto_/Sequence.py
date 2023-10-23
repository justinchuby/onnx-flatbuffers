# automatically generated by the FlatBuffers compiler, do not modify

# namespace: TypeProto_

import flatbuffers
from flatbuffers.compat import import_numpy
np = import_numpy()

class Sequence(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAs(cls, buf, offset=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = Sequence()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def GetRootAsSequence(cls, buf, offset=0):
        """This method is deprecated. Please switch to GetRootAs."""
        return cls.GetRootAs(buf, offset)
    # Sequence
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # Sequence
    def ElemType(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            x = self._tab.Indirect(o + self._tab.Pos)
            from onnxfbs.protos.TypeProto import TypeProto
            obj = TypeProto()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

def SequenceStart(builder):
    builder.StartObject(1)

def Start(builder):
    SequenceStart(builder)

def SequenceAddElemType(builder, elemType):
    builder.PrependUOffsetTRelativeSlot(0, flatbuffers.number_types.UOffsetTFlags.py_type(elemType), 0)

def AddElemType(builder, elemType):
    SequenceAddElemType(builder, elemType)

def SequenceEnd(builder):
    return builder.EndObject()

def End(builder):
    return SequenceEnd(builder)

import onnx.TypeProto
try:
    from typing import Optional
except:
    pass

class SequenceT(object):

    # SequenceT
    def __init__(self):
        self.elemType = None  # type: Optional[onnx.TypeProto.TypeProtoT]

    @classmethod
    def InitFromBuf(cls, buf, pos):
        sequence = Sequence()
        sequence.Init(buf, pos)
        return cls.InitFromObj(sequence)

    @classmethod
    def InitFromPackedBuf(cls, buf, pos=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, pos)
        return cls.InitFromBuf(buf, pos+n)

    @classmethod
    def InitFromObj(cls, sequence):
        x = SequenceT()
        x._UnPack(sequence)
        return x

    # SequenceT
    def _UnPack(self, sequence):
        if sequence is None:
            return
        if sequence.ElemType() is not None:
            self.elemType = onnx.TypeProto.TypeProtoT.InitFromObj(sequence.ElemType())

    # SequenceT
    def Pack(self, builder):
        if self.elemType is not None:
            elemType = self.elemType.Pack(builder)
        SequenceStart(builder)
        if self.elemType is not None:
            SequenceAddElemType(builder, elemType)
        sequence = SequenceEnd(builder)
        return sequence
