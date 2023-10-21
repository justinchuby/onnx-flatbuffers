# automatically generated by the FlatBuffers compiler, do not modify

# namespace: onnx

import flatbuffers
from flatbuffers.compat import import_numpy
np = import_numpy()

class AttributeProto(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAs(cls, buf, offset=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = AttributeProto()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def GetRootAsAttributeProto(cls, buf, offset=0):
        """This method is deprecated. Please switch to GetRootAs."""
        return cls.GetRootAs(buf, offset)
    # AttributeProto
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # AttributeProto
    def Name(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.String(o + self._tab.Pos)
        return None

    # AttributeProto
    def RefAttrName(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.String(o + self._tab.Pos)
        return None

    # AttributeProto
    def DocString(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            return self._tab.String(o + self._tab.Pos)
        return None

    # AttributeProto
    def Type(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(10))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 0

    # AttributeProto
    def F(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(12))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Float32Flags, o + self._tab.Pos)
        return 0.0

    # AttributeProto
    def I(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(14))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int64Flags, o + self._tab.Pos)
        return 0

    # AttributeProto
    def S(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(16))
        if o != 0:
            a = self._tab.Vector(o)
            return self._tab.Get(flatbuffers.number_types.Uint8Flags, a + flatbuffers.number_types.UOffsetTFlags.py_type(j * 1))
        return 0

    # AttributeProto
    def SAsNumpy(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(16))
        if o != 0:
            return self._tab.GetVectorAsNumpy(flatbuffers.number_types.Uint8Flags, o)
        return 0

    # AttributeProto
    def SLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(16))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # AttributeProto
    def SIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(16))
        return o == 0

    # AttributeProto
    def T(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(18))
        if o != 0:
            x = self._tab.Indirect(o + self._tab.Pos)
            from onnx.TensorProto import TensorProto
            obj = TensorProto()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

    # AttributeProto
    def G(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(20))
        if o != 0:
            x = self._tab.Indirect(o + self._tab.Pos)
            from onnx.GraphProto import GraphProto
            obj = GraphProto()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

    # AttributeProto
    def SparseTensor(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(22))
        if o != 0:
            x = self._tab.Indirect(o + self._tab.Pos)
            from onnx.SparseTensorProto import SparseTensorProto
            obj = SparseTensorProto()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

    # AttributeProto
    def Tp(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(24))
        if o != 0:
            x = self._tab.Indirect(o + self._tab.Pos)
            from onnx.TypeProto import TypeProto
            obj = TypeProto()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

    # AttributeProto
    def Floats(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(26))
        if o != 0:
            a = self._tab.Vector(o)
            return self._tab.Get(flatbuffers.number_types.Float32Flags, a + flatbuffers.number_types.UOffsetTFlags.py_type(j * 4))
        return 0

    # AttributeProto
    def FloatsAsNumpy(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(26))
        if o != 0:
            return self._tab.GetVectorAsNumpy(flatbuffers.number_types.Float32Flags, o)
        return 0

    # AttributeProto
    def FloatsLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(26))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # AttributeProto
    def FloatsIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(26))
        return o == 0

    # AttributeProto
    def Ints(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(28))
        if o != 0:
            a = self._tab.Vector(o)
            return self._tab.Get(flatbuffers.number_types.Int64Flags, a + flatbuffers.number_types.UOffsetTFlags.py_type(j * 8))
        return 0

    # AttributeProto
    def IntsAsNumpy(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(28))
        if o != 0:
            return self._tab.GetVectorAsNumpy(flatbuffers.number_types.Int64Flags, o)
        return 0

    # AttributeProto
    def IntsLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(28))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # AttributeProto
    def IntsIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(28))
        return o == 0

    # AttributeProto
    def Strings(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(30))
        if o != 0:
            a = self._tab.Vector(o)
            return self._tab.String(a + flatbuffers.number_types.UOffsetTFlags.py_type(j * 4))
        return ""

    # AttributeProto
    def StringsLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(30))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # AttributeProto
    def StringsIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(30))
        return o == 0

    # AttributeProto
    def Tensors(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(32))
        if o != 0:
            x = self._tab.Vector(o)
            x += flatbuffers.number_types.UOffsetTFlags.py_type(j) * 4
            x = self._tab.Indirect(x)
            from onnx.TensorProto import TensorProto
            obj = TensorProto()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

    # AttributeProto
    def TensorsLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(32))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # AttributeProto
    def TensorsIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(32))
        return o == 0

    # AttributeProto
    def Graphs(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(34))
        if o != 0:
            x = self._tab.Vector(o)
            x += flatbuffers.number_types.UOffsetTFlags.py_type(j) * 4
            x = self._tab.Indirect(x)
            from onnx.GraphProto import GraphProto
            obj = GraphProto()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

    # AttributeProto
    def GraphsLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(34))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # AttributeProto
    def GraphsIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(34))
        return o == 0

    # AttributeProto
    def SparseTensors(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(36))
        if o != 0:
            x = self._tab.Vector(o)
            x += flatbuffers.number_types.UOffsetTFlags.py_type(j) * 4
            x = self._tab.Indirect(x)
            from onnx.SparseTensorProto import SparseTensorProto
            obj = SparseTensorProto()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

    # AttributeProto
    def SparseTensorsLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(36))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # AttributeProto
    def SparseTensorsIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(36))
        return o == 0

    # AttributeProto
    def TypeProtos(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(38))
        if o != 0:
            x = self._tab.Vector(o)
            x += flatbuffers.number_types.UOffsetTFlags.py_type(j) * 4
            x = self._tab.Indirect(x)
            from onnx.TypeProto import TypeProto
            obj = TypeProto()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

    # AttributeProto
    def TypeProtosLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(38))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # AttributeProto
    def TypeProtosIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(38))
        return o == 0

def AttributeProtoStart(builder):
    builder.StartObject(18)

def Start(builder):
    AttributeProtoStart(builder)

def AttributeProtoAddName(builder, name):
    builder.PrependUOffsetTRelativeSlot(0, flatbuffers.number_types.UOffsetTFlags.py_type(name), 0)

def AddName(builder, name):
    AttributeProtoAddName(builder, name)

def AttributeProtoAddRefAttrName(builder, refAttrName):
    builder.PrependUOffsetTRelativeSlot(1, flatbuffers.number_types.UOffsetTFlags.py_type(refAttrName), 0)

def AddRefAttrName(builder, refAttrName):
    AttributeProtoAddRefAttrName(builder, refAttrName)

def AttributeProtoAddDocString(builder, docString):
    builder.PrependUOffsetTRelativeSlot(2, flatbuffers.number_types.UOffsetTFlags.py_type(docString), 0)

def AddDocString(builder, docString):
    AttributeProtoAddDocString(builder, docString)

def AttributeProtoAddType(builder, type):
    builder.PrependInt32Slot(3, type, 0)

def AddType(builder, type):
    AttributeProtoAddType(builder, type)

def AttributeProtoAddF(builder, f):
    builder.PrependFloat32Slot(4, f, 0.0)

def AddF(builder, f):
    AttributeProtoAddF(builder, f)

def AttributeProtoAddI(builder, i):
    builder.PrependInt64Slot(5, i, 0)

def AddI(builder, i):
    AttributeProtoAddI(builder, i)

def AttributeProtoAddS(builder, s):
    builder.PrependUOffsetTRelativeSlot(6, flatbuffers.number_types.UOffsetTFlags.py_type(s), 0)

def AddS(builder, s):
    AttributeProtoAddS(builder, s)

def AttributeProtoStartSVector(builder, numElems):
    return builder.StartVector(1, numElems, 1)

def StartSVector(builder, numElems: int) -> int:
    return AttributeProtoStartSVector(builder, numElems)

def AttributeProtoAddT(builder, t):
    builder.PrependUOffsetTRelativeSlot(7, flatbuffers.number_types.UOffsetTFlags.py_type(t), 0)

def AddT(builder, t):
    AttributeProtoAddT(builder, t)

def AttributeProtoAddG(builder, g):
    builder.PrependUOffsetTRelativeSlot(8, flatbuffers.number_types.UOffsetTFlags.py_type(g), 0)

def AddG(builder, g):
    AttributeProtoAddG(builder, g)

def AttributeProtoAddSparseTensor(builder, sparseTensor):
    builder.PrependUOffsetTRelativeSlot(9, flatbuffers.number_types.UOffsetTFlags.py_type(sparseTensor), 0)

def AddSparseTensor(builder, sparseTensor):
    AttributeProtoAddSparseTensor(builder, sparseTensor)

def AttributeProtoAddTp(builder, tp):
    builder.PrependUOffsetTRelativeSlot(10, flatbuffers.number_types.UOffsetTFlags.py_type(tp), 0)

def AddTp(builder, tp):
    AttributeProtoAddTp(builder, tp)

def AttributeProtoAddFloats(builder, floats):
    builder.PrependUOffsetTRelativeSlot(11, flatbuffers.number_types.UOffsetTFlags.py_type(floats), 0)

def AddFloats(builder, floats):
    AttributeProtoAddFloats(builder, floats)

def AttributeProtoStartFloatsVector(builder, numElems):
    return builder.StartVector(4, numElems, 4)

def StartFloatsVector(builder, numElems: int) -> int:
    return AttributeProtoStartFloatsVector(builder, numElems)

def AttributeProtoAddInts(builder, ints):
    builder.PrependUOffsetTRelativeSlot(12, flatbuffers.number_types.UOffsetTFlags.py_type(ints), 0)

def AddInts(builder, ints):
    AttributeProtoAddInts(builder, ints)

def AttributeProtoStartIntsVector(builder, numElems):
    return builder.StartVector(8, numElems, 8)

def StartIntsVector(builder, numElems: int) -> int:
    return AttributeProtoStartIntsVector(builder, numElems)

def AttributeProtoAddStrings(builder, strings):
    builder.PrependUOffsetTRelativeSlot(13, flatbuffers.number_types.UOffsetTFlags.py_type(strings), 0)

def AddStrings(builder, strings):
    AttributeProtoAddStrings(builder, strings)

def AttributeProtoStartStringsVector(builder, numElems):
    return builder.StartVector(4, numElems, 4)

def StartStringsVector(builder, numElems: int) -> int:
    return AttributeProtoStartStringsVector(builder, numElems)

def AttributeProtoAddTensors(builder, tensors):
    builder.PrependUOffsetTRelativeSlot(14, flatbuffers.number_types.UOffsetTFlags.py_type(tensors), 0)

def AddTensors(builder, tensors):
    AttributeProtoAddTensors(builder, tensors)

def AttributeProtoStartTensorsVector(builder, numElems):
    return builder.StartVector(4, numElems, 4)

def StartTensorsVector(builder, numElems: int) -> int:
    return AttributeProtoStartTensorsVector(builder, numElems)

def AttributeProtoAddGraphs(builder, graphs):
    builder.PrependUOffsetTRelativeSlot(15, flatbuffers.number_types.UOffsetTFlags.py_type(graphs), 0)

def AddGraphs(builder, graphs):
    AttributeProtoAddGraphs(builder, graphs)

def AttributeProtoStartGraphsVector(builder, numElems):
    return builder.StartVector(4, numElems, 4)

def StartGraphsVector(builder, numElems: int) -> int:
    return AttributeProtoStartGraphsVector(builder, numElems)

def AttributeProtoAddSparseTensors(builder, sparseTensors):
    builder.PrependUOffsetTRelativeSlot(16, flatbuffers.number_types.UOffsetTFlags.py_type(sparseTensors), 0)

def AddSparseTensors(builder, sparseTensors):
    AttributeProtoAddSparseTensors(builder, sparseTensors)

def AttributeProtoStartSparseTensorsVector(builder, numElems):
    return builder.StartVector(4, numElems, 4)

def StartSparseTensorsVector(builder, numElems: int) -> int:
    return AttributeProtoStartSparseTensorsVector(builder, numElems)

def AttributeProtoAddTypeProtos(builder, typeProtos):
    builder.PrependUOffsetTRelativeSlot(17, flatbuffers.number_types.UOffsetTFlags.py_type(typeProtos), 0)

def AddTypeProtos(builder, typeProtos):
    AttributeProtoAddTypeProtos(builder, typeProtos)

def AttributeProtoStartTypeProtosVector(builder, numElems):
    return builder.StartVector(4, numElems, 4)

def StartTypeProtosVector(builder, numElems: int) -> int:
    return AttributeProtoStartTypeProtosVector(builder, numElems)

def AttributeProtoEnd(builder):
    return builder.EndObject()

def End(builder):
    return AttributeProtoEnd(builder)
