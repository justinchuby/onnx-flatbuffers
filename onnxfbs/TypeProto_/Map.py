# automatically generated by the FlatBuffers compiler, do not modify

# namespace: TypeProto_

import flatbuffers
from flatbuffers.compat import import_numpy
np = import_numpy()

class Map(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAs(cls, buf, offset=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = Map()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def GetRootAsMap(cls, buf, offset=0):
        """This method is deprecated. Please switch to GetRootAs."""
        return cls.GetRootAs(buf, offset)
    # Map
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # Map
    def KeyType(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 0

    # Map
    def ValueType(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            x = self._tab.Indirect(o + self._tab.Pos)
            from onnx.TypeProto import TypeProto
            obj = TypeProto()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

def MapStart(builder):
    builder.StartObject(2)

def Start(builder):
    MapStart(builder)

def MapAddKeyType(builder, keyType):
    builder.PrependInt32Slot(0, keyType, 0)

def AddKeyType(builder, keyType):
    MapAddKeyType(builder, keyType)

def MapAddValueType(builder, valueType):
    builder.PrependUOffsetTRelativeSlot(1, flatbuffers.number_types.UOffsetTFlags.py_type(valueType), 0)

def AddValueType(builder, valueType):
    MapAddValueType(builder, valueType)

def MapEnd(builder):
    return builder.EndObject()

def End(builder):
    return MapEnd(builder)

import onnx.TypeProto
try:
    from typing import Optional
except:
    pass

class MapT(object):

    # MapT
    def __init__(self):
        self.keyType = 0  # type: int
        self.valueType = None  # type: Optional[onnx.TypeProto.TypeProtoT]

    @classmethod
    def InitFromBuf(cls, buf, pos):
        map = Map()
        map.Init(buf, pos)
        return cls.InitFromObj(map)

    @classmethod
    def InitFromPackedBuf(cls, buf, pos=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, pos)
        return cls.InitFromBuf(buf, pos+n)

    @classmethod
    def InitFromObj(cls, map):
        x = MapT()
        x._UnPack(map)
        return x

    # MapT
    def _UnPack(self, map):
        if map is None:
            return
        self.keyType = map.KeyType()
        if map.ValueType() is not None:
            self.valueType = onnx.TypeProto.TypeProtoT.InitFromObj(map.ValueType())

    # MapT
    def Pack(self, builder):
        if self.valueType is not None:
            valueType = self.valueType.Pack(builder)
        MapStart(builder)
        MapAddKeyType(builder, self.keyType)
        if self.valueType is not None:
            MapAddValueType(builder, valueType)
        map = MapEnd(builder)
        return map
