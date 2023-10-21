# automatically generated by the FlatBuffers compiler, do not modify

# namespace: onnx

import flatbuffers
from flatbuffers.compat import import_numpy
np = import_numpy()

class TensorProto(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAs(cls, buf, offset=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = TensorProto()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def GetRootAsTensorProto(cls, buf, offset=0):
        """This method is deprecated. Please switch to GetRootAs."""
        return cls.GetRootAs(buf, offset)
    # TensorProto
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # TensorProto
    def Dims(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            a = self._tab.Vector(o)
            return self._tab.Get(flatbuffers.number_types.Int64Flags, a + flatbuffers.number_types.UOffsetTFlags.py_type(j * 8))
        return 0

    # TensorProto
    def DimsAsNumpy(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.GetVectorAsNumpy(flatbuffers.number_types.Int64Flags, o)
        return 0

    # TensorProto
    def DimsLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # TensorProto
    def DimsIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        return o == 0

    # TensorProto
    def DataType(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 0

    # TensorProto
    def Segment(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            x = self._tab.Indirect(o + self._tab.Pos)
            from onnx.TensorProto_.Segment import Segment
            obj = Segment()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

    # TensorProto
    def FloatData(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(10))
        if o != 0:
            a = self._tab.Vector(o)
            return self._tab.Get(flatbuffers.number_types.Float32Flags, a + flatbuffers.number_types.UOffsetTFlags.py_type(j * 4))
        return 0

    # TensorProto
    def FloatDataAsNumpy(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(10))
        if o != 0:
            return self._tab.GetVectorAsNumpy(flatbuffers.number_types.Float32Flags, o)
        return 0

    # TensorProto
    def FloatDataLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(10))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # TensorProto
    def FloatDataIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(10))
        return o == 0

    # TensorProto
    def Int32Data(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(12))
        if o != 0:
            a = self._tab.Vector(o)
            return self._tab.Get(flatbuffers.number_types.Int32Flags, a + flatbuffers.number_types.UOffsetTFlags.py_type(j * 4))
        return 0

    # TensorProto
    def Int32DataAsNumpy(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(12))
        if o != 0:
            return self._tab.GetVectorAsNumpy(flatbuffers.number_types.Int32Flags, o)
        return 0

    # TensorProto
    def Int32DataLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(12))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # TensorProto
    def Int32DataIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(12))
        return o == 0

    # TensorProto
    def StringData(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(14))
        if o != 0:
            a = self._tab.Vector(o)
            return self._tab.String(a + flatbuffers.number_types.UOffsetTFlags.py_type(j * 4))
        return ""

    # TensorProto
    def StringDataLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(14))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # TensorProto
    def StringDataIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(14))
        return o == 0

    # TensorProto
    def Int64Data(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(16))
        if o != 0:
            a = self._tab.Vector(o)
            return self._tab.Get(flatbuffers.number_types.Int64Flags, a + flatbuffers.number_types.UOffsetTFlags.py_type(j * 8))
        return 0

    # TensorProto
    def Int64DataAsNumpy(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(16))
        if o != 0:
            return self._tab.GetVectorAsNumpy(flatbuffers.number_types.Int64Flags, o)
        return 0

    # TensorProto
    def Int64DataLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(16))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # TensorProto
    def Int64DataIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(16))
        return o == 0

    # TensorProto
    def Name(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(18))
        if o != 0:
            return self._tab.String(o + self._tab.Pos)
        return None

    # TensorProto
    def DocString(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(20))
        if o != 0:
            return self._tab.String(o + self._tab.Pos)
        return None

    # TensorProto
    def RawData(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(22))
        if o != 0:
            a = self._tab.Vector(o)
            return self._tab.Get(flatbuffers.number_types.Uint8Flags, a + flatbuffers.number_types.UOffsetTFlags.py_type(j * 1))
        return 0

    # TensorProto
    def RawDataAsNumpy(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(22))
        if o != 0:
            return self._tab.GetVectorAsNumpy(flatbuffers.number_types.Uint8Flags, o)
        return 0

    # TensorProto
    def RawDataLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(22))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # TensorProto
    def RawDataIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(22))
        return o == 0

    # TensorProto
    def ExternalData(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(24))
        if o != 0:
            x = self._tab.Vector(o)
            x += flatbuffers.number_types.UOffsetTFlags.py_type(j) * 4
            x = self._tab.Indirect(x)
            from onnx.StringStringEntryProto import StringStringEntryProto
            obj = StringStringEntryProto()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

    # TensorProto
    def ExternalDataLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(24))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # TensorProto
    def ExternalDataIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(24))
        return o == 0

    # TensorProto
    def DataLocation(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(26))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 0

    # TensorProto
    def DoubleData(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(28))
        if o != 0:
            a = self._tab.Vector(o)
            return self._tab.Get(flatbuffers.number_types.Float64Flags, a + flatbuffers.number_types.UOffsetTFlags.py_type(j * 8))
        return 0

    # TensorProto
    def DoubleDataAsNumpy(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(28))
        if o != 0:
            return self._tab.GetVectorAsNumpy(flatbuffers.number_types.Float64Flags, o)
        return 0

    # TensorProto
    def DoubleDataLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(28))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # TensorProto
    def DoubleDataIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(28))
        return o == 0

    # TensorProto
    def Uint64Data(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(30))
        if o != 0:
            a = self._tab.Vector(o)
            return self._tab.Get(flatbuffers.number_types.Uint64Flags, a + flatbuffers.number_types.UOffsetTFlags.py_type(j * 8))
        return 0

    # TensorProto
    def Uint64DataAsNumpy(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(30))
        if o != 0:
            return self._tab.GetVectorAsNumpy(flatbuffers.number_types.Uint64Flags, o)
        return 0

    # TensorProto
    def Uint64DataLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(30))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # TensorProto
    def Uint64DataIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(30))
        return o == 0

def TensorProtoStart(builder):
    builder.StartObject(14)

def Start(builder):
    TensorProtoStart(builder)

def TensorProtoAddDims(builder, dims):
    builder.PrependUOffsetTRelativeSlot(0, flatbuffers.number_types.UOffsetTFlags.py_type(dims), 0)

def AddDims(builder, dims):
    TensorProtoAddDims(builder, dims)

def TensorProtoStartDimsVector(builder, numElems):
    return builder.StartVector(8, numElems, 8)

def StartDimsVector(builder, numElems: int) -> int:
    return TensorProtoStartDimsVector(builder, numElems)

def TensorProtoAddDataType(builder, dataType):
    builder.PrependInt32Slot(1, dataType, 0)

def AddDataType(builder, dataType):
    TensorProtoAddDataType(builder, dataType)

def TensorProtoAddSegment(builder, segment):
    builder.PrependUOffsetTRelativeSlot(2, flatbuffers.number_types.UOffsetTFlags.py_type(segment), 0)

def AddSegment(builder, segment):
    TensorProtoAddSegment(builder, segment)

def TensorProtoAddFloatData(builder, floatData):
    builder.PrependUOffsetTRelativeSlot(3, flatbuffers.number_types.UOffsetTFlags.py_type(floatData), 0)

def AddFloatData(builder, floatData):
    TensorProtoAddFloatData(builder, floatData)

def TensorProtoStartFloatDataVector(builder, numElems):
    return builder.StartVector(4, numElems, 4)

def StartFloatDataVector(builder, numElems: int) -> int:
    return TensorProtoStartFloatDataVector(builder, numElems)

def TensorProtoAddInt32Data(builder, int32Data):
    builder.PrependUOffsetTRelativeSlot(4, flatbuffers.number_types.UOffsetTFlags.py_type(int32Data), 0)

def AddInt32Data(builder, int32Data):
    TensorProtoAddInt32Data(builder, int32Data)

def TensorProtoStartInt32DataVector(builder, numElems):
    return builder.StartVector(4, numElems, 4)

def StartInt32DataVector(builder, numElems: int) -> int:
    return TensorProtoStartInt32DataVector(builder, numElems)

def TensorProtoAddStringData(builder, stringData):
    builder.PrependUOffsetTRelativeSlot(5, flatbuffers.number_types.UOffsetTFlags.py_type(stringData), 0)

def AddStringData(builder, stringData):
    TensorProtoAddStringData(builder, stringData)

def TensorProtoStartStringDataVector(builder, numElems):
    return builder.StartVector(4, numElems, 4)

def StartStringDataVector(builder, numElems: int) -> int:
    return TensorProtoStartStringDataVector(builder, numElems)

def TensorProtoAddInt64Data(builder, int64Data):
    builder.PrependUOffsetTRelativeSlot(6, flatbuffers.number_types.UOffsetTFlags.py_type(int64Data), 0)

def AddInt64Data(builder, int64Data):
    TensorProtoAddInt64Data(builder, int64Data)

def TensorProtoStartInt64DataVector(builder, numElems):
    return builder.StartVector(8, numElems, 8)

def StartInt64DataVector(builder, numElems: int) -> int:
    return TensorProtoStartInt64DataVector(builder, numElems)

def TensorProtoAddName(builder, name):
    builder.PrependUOffsetTRelativeSlot(7, flatbuffers.number_types.UOffsetTFlags.py_type(name), 0)

def AddName(builder, name):
    TensorProtoAddName(builder, name)

def TensorProtoAddDocString(builder, docString):
    builder.PrependUOffsetTRelativeSlot(8, flatbuffers.number_types.UOffsetTFlags.py_type(docString), 0)

def AddDocString(builder, docString):
    TensorProtoAddDocString(builder, docString)

def TensorProtoAddRawData(builder, rawData):
    builder.PrependUOffsetTRelativeSlot(9, flatbuffers.number_types.UOffsetTFlags.py_type(rawData), 0)

def AddRawData(builder, rawData):
    TensorProtoAddRawData(builder, rawData)

def TensorProtoStartRawDataVector(builder, numElems):
    return builder.StartVector(1, numElems, 1)

def StartRawDataVector(builder, numElems: int) -> int:
    return TensorProtoStartRawDataVector(builder, numElems)

def TensorProtoAddExternalData(builder, externalData):
    builder.PrependUOffsetTRelativeSlot(10, flatbuffers.number_types.UOffsetTFlags.py_type(externalData), 0)

def AddExternalData(builder, externalData):
    TensorProtoAddExternalData(builder, externalData)

def TensorProtoStartExternalDataVector(builder, numElems):
    return builder.StartVector(4, numElems, 4)

def StartExternalDataVector(builder, numElems: int) -> int:
    return TensorProtoStartExternalDataVector(builder, numElems)

def TensorProtoAddDataLocation(builder, dataLocation):
    builder.PrependInt32Slot(11, dataLocation, 0)

def AddDataLocation(builder, dataLocation):
    TensorProtoAddDataLocation(builder, dataLocation)

def TensorProtoAddDoubleData(builder, doubleData):
    builder.PrependUOffsetTRelativeSlot(12, flatbuffers.number_types.UOffsetTFlags.py_type(doubleData), 0)

def AddDoubleData(builder, doubleData):
    TensorProtoAddDoubleData(builder, doubleData)

def TensorProtoStartDoubleDataVector(builder, numElems):
    return builder.StartVector(8, numElems, 8)

def StartDoubleDataVector(builder, numElems: int) -> int:
    return TensorProtoStartDoubleDataVector(builder, numElems)

def TensorProtoAddUint64Data(builder, uint64Data):
    builder.PrependUOffsetTRelativeSlot(13, flatbuffers.number_types.UOffsetTFlags.py_type(uint64Data), 0)

def AddUint64Data(builder, uint64Data):
    TensorProtoAddUint64Data(builder, uint64Data)

def TensorProtoStartUint64DataVector(builder, numElems):
    return builder.StartVector(8, numElems, 8)

def StartUint64DataVector(builder, numElems: int) -> int:
    return TensorProtoStartUint64DataVector(builder, numElems)

def TensorProtoEnd(builder):
    return builder.EndObject()

def End(builder):
    return TensorProtoEnd(builder)
