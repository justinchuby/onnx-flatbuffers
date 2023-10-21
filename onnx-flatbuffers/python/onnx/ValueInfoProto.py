# automatically generated by the FlatBuffers compiler, do not modify

# namespace: onnx

import flatbuffers
from flatbuffers.compat import import_numpy
np = import_numpy()

class ValueInfoProto(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAs(cls, buf, offset=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = ValueInfoProto()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def GetRootAsValueInfoProto(cls, buf, offset=0):
        """This method is deprecated. Please switch to GetRootAs."""
        return cls.GetRootAs(buf, offset)
    # ValueInfoProto
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # ValueInfoProto
    def Name(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.String(o + self._tab.Pos)
        return None

    # ValueInfoProto
    def Type(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            x = self._tab.Indirect(o + self._tab.Pos)
            from onnx.TypeProto import TypeProto
            obj = TypeProto()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

    # ValueInfoProto
    def DocString(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            return self._tab.String(o + self._tab.Pos)
        return None

def ValueInfoProtoStart(builder):
    builder.StartObject(3)

def Start(builder):
    ValueInfoProtoStart(builder)

def ValueInfoProtoAddName(builder, name):
    builder.PrependUOffsetTRelativeSlot(0, flatbuffers.number_types.UOffsetTFlags.py_type(name), 0)

def AddName(builder, name):
    ValueInfoProtoAddName(builder, name)

def ValueInfoProtoAddType(builder, type):
    builder.PrependUOffsetTRelativeSlot(1, flatbuffers.number_types.UOffsetTFlags.py_type(type), 0)

def AddType(builder, type):
    ValueInfoProtoAddType(builder, type)

def ValueInfoProtoAddDocString(builder, docString):
    builder.PrependUOffsetTRelativeSlot(2, flatbuffers.number_types.UOffsetTFlags.py_type(docString), 0)

def AddDocString(builder, docString):
    ValueInfoProtoAddDocString(builder, docString)

def ValueInfoProtoEnd(builder):
    return builder.EndObject()

def End(builder):
    return ValueInfoProtoEnd(builder)