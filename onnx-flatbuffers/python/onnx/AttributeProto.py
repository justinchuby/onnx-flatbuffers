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

import onnx.GraphProto
import onnx.SparseTensorProto
import onnx.TensorProto
import onnx.TypeProto
try:
    from typing import List, Optional
except:
    pass

class AttributeProtoT(object):

    # AttributeProtoT
    def __init__(self):
        self.name = None  # type: str
        self.refAttrName = None  # type: str
        self.docString = None  # type: str
        self.type = 0  # type: int
        self.f = 0.0  # type: float
        self.i = 0  # type: int
        self.s = None  # type: List[int]
        self.t = None  # type: Optional[onnx.TensorProto.TensorProtoT]
        self.g = None  # type: Optional[onnx.GraphProto.GraphProtoT]
        self.sparseTensor = None  # type: Optional[onnx.SparseTensorProto.SparseTensorProtoT]
        self.tp = None  # type: Optional[onnx.TypeProto.TypeProtoT]
        self.floats = None  # type: List[float]
        self.ints = None  # type: List[int]
        self.strings = None  # type: List[str]
        self.tensors = None  # type: List[onnx.TensorProto.TensorProtoT]
        self.graphs = None  # type: List[onnx.GraphProto.GraphProtoT]
        self.sparseTensors = None  # type: List[onnx.SparseTensorProto.SparseTensorProtoT]
        self.typeProtos = None  # type: List[onnx.TypeProto.TypeProtoT]

    @classmethod
    def InitFromBuf(cls, buf, pos):
        attributeProto = AttributeProto()
        attributeProto.Init(buf, pos)
        return cls.InitFromObj(attributeProto)

    @classmethod
    def InitFromPackedBuf(cls, buf, pos=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, pos)
        return cls.InitFromBuf(buf, pos+n)

    @classmethod
    def InitFromObj(cls, attributeProto):
        x = AttributeProtoT()
        x._UnPack(attributeProto)
        return x

    # AttributeProtoT
    def _UnPack(self, attributeProto):
        if attributeProto is None:
            return
        self.name = attributeProto.Name()
        self.refAttrName = attributeProto.RefAttrName()
        self.docString = attributeProto.DocString()
        self.type = attributeProto.Type()
        self.f = attributeProto.F()
        self.i = attributeProto.I()
        if not attributeProto.SIsNone():
            if np is None:
                self.s = []
                for i in range(attributeProto.SLength()):
                    self.s.append(attributeProto.S(i))
            else:
                self.s = attributeProto.SAsNumpy()
        if attributeProto.T() is not None:
            self.t = onnx.TensorProto.TensorProtoT.InitFromObj(attributeProto.T())
        if attributeProto.G() is not None:
            self.g = onnx.GraphProto.GraphProtoT.InitFromObj(attributeProto.G())
        if attributeProto.SparseTensor() is not None:
            self.sparseTensor = onnx.SparseTensorProto.SparseTensorProtoT.InitFromObj(attributeProto.SparseTensor())
        if attributeProto.Tp() is not None:
            self.tp = onnx.TypeProto.TypeProtoT.InitFromObj(attributeProto.Tp())
        if not attributeProto.FloatsIsNone():
            if np is None:
                self.floats = []
                for i in range(attributeProto.FloatsLength()):
                    self.floats.append(attributeProto.Floats(i))
            else:
                self.floats = attributeProto.FloatsAsNumpy()
        if not attributeProto.IntsIsNone():
            if np is None:
                self.ints = []
                for i in range(attributeProto.IntsLength()):
                    self.ints.append(attributeProto.Ints(i))
            else:
                self.ints = attributeProto.IntsAsNumpy()
        if not attributeProto.StringsIsNone():
            self.strings = []
            for i in range(attributeProto.StringsLength()):
                self.strings.append(attributeProto.Strings(i))
        if not attributeProto.TensorsIsNone():
            self.tensors = []
            for i in range(attributeProto.TensorsLength()):
                if attributeProto.Tensors(i) is None:
                    self.tensors.append(None)
                else:
                    tensorProto_ = onnx.TensorProto.TensorProtoT.InitFromObj(attributeProto.Tensors(i))
                    self.tensors.append(tensorProto_)
        if not attributeProto.GraphsIsNone():
            self.graphs = []
            for i in range(attributeProto.GraphsLength()):
                if attributeProto.Graphs(i) is None:
                    self.graphs.append(None)
                else:
                    graphProto_ = onnx.GraphProto.GraphProtoT.InitFromObj(attributeProto.Graphs(i))
                    self.graphs.append(graphProto_)
        if not attributeProto.SparseTensorsIsNone():
            self.sparseTensors = []
            for i in range(attributeProto.SparseTensorsLength()):
                if attributeProto.SparseTensors(i) is None:
                    self.sparseTensors.append(None)
                else:
                    sparseTensorProto_ = onnx.SparseTensorProto.SparseTensorProtoT.InitFromObj(attributeProto.SparseTensors(i))
                    self.sparseTensors.append(sparseTensorProto_)
        if not attributeProto.TypeProtosIsNone():
            self.typeProtos = []
            for i in range(attributeProto.TypeProtosLength()):
                if attributeProto.TypeProtos(i) is None:
                    self.typeProtos.append(None)
                else:
                    typeProto_ = onnx.TypeProto.TypeProtoT.InitFromObj(attributeProto.TypeProtos(i))
                    self.typeProtos.append(typeProto_)

    # AttributeProtoT
    def Pack(self, builder):
        if self.name is not None:
            name = builder.CreateString(self.name)
        if self.refAttrName is not None:
            refAttrName = builder.CreateString(self.refAttrName)
        if self.docString is not None:
            docString = builder.CreateString(self.docString)
        if self.s is not None:
            if np is not None and type(self.s) is np.ndarray:
                s = builder.CreateNumpyVector(self.s)
            else:
                AttributeProtoStartSVector(builder, len(self.s))
                for i in reversed(range(len(self.s))):
                    builder.PrependUint8(self.s[i])
                s = builder.EndVector()
        if self.t is not None:
            t = self.t.Pack(builder)
        if self.g is not None:
            g = self.g.Pack(builder)
        if self.sparseTensor is not None:
            sparseTensor = self.sparseTensor.Pack(builder)
        if self.tp is not None:
            tp = self.tp.Pack(builder)
        if self.floats is not None:
            if np is not None and type(self.floats) is np.ndarray:
                floats = builder.CreateNumpyVector(self.floats)
            else:
                AttributeProtoStartFloatsVector(builder, len(self.floats))
                for i in reversed(range(len(self.floats))):
                    builder.PrependFloat32(self.floats[i])
                floats = builder.EndVector()
        if self.ints is not None:
            if np is not None and type(self.ints) is np.ndarray:
                ints = builder.CreateNumpyVector(self.ints)
            else:
                AttributeProtoStartIntsVector(builder, len(self.ints))
                for i in reversed(range(len(self.ints))):
                    builder.PrependInt64(self.ints[i])
                ints = builder.EndVector()
        if self.strings is not None:
            stringslist = []
            for i in range(len(self.strings)):
                stringslist.append(builder.CreateString(self.strings[i]))
            AttributeProtoStartStringsVector(builder, len(self.strings))
            for i in reversed(range(len(self.strings))):
                builder.PrependUOffsetTRelative(stringslist[i])
            strings = builder.EndVector()
        if self.tensors is not None:
            tensorslist = []
            for i in range(len(self.tensors)):
                tensorslist.append(self.tensors[i].Pack(builder))
            AttributeProtoStartTensorsVector(builder, len(self.tensors))
            for i in reversed(range(len(self.tensors))):
                builder.PrependUOffsetTRelative(tensorslist[i])
            tensors = builder.EndVector()
        if self.graphs is not None:
            graphslist = []
            for i in range(len(self.graphs)):
                graphslist.append(self.graphs[i].Pack(builder))
            AttributeProtoStartGraphsVector(builder, len(self.graphs))
            for i in reversed(range(len(self.graphs))):
                builder.PrependUOffsetTRelative(graphslist[i])
            graphs = builder.EndVector()
        if self.sparseTensors is not None:
            sparseTensorslist = []
            for i in range(len(self.sparseTensors)):
                sparseTensorslist.append(self.sparseTensors[i].Pack(builder))
            AttributeProtoStartSparseTensorsVector(builder, len(self.sparseTensors))
            for i in reversed(range(len(self.sparseTensors))):
                builder.PrependUOffsetTRelative(sparseTensorslist[i])
            sparseTensors = builder.EndVector()
        if self.typeProtos is not None:
            typeProtoslist = []
            for i in range(len(self.typeProtos)):
                typeProtoslist.append(self.typeProtos[i].Pack(builder))
            AttributeProtoStartTypeProtosVector(builder, len(self.typeProtos))
            for i in reversed(range(len(self.typeProtos))):
                builder.PrependUOffsetTRelative(typeProtoslist[i])
            typeProtos = builder.EndVector()
        AttributeProtoStart(builder)
        if self.name is not None:
            AttributeProtoAddName(builder, name)
        if self.refAttrName is not None:
            AttributeProtoAddRefAttrName(builder, refAttrName)
        if self.docString is not None:
            AttributeProtoAddDocString(builder, docString)
        AttributeProtoAddType(builder, self.type)
        AttributeProtoAddF(builder, self.f)
        AttributeProtoAddI(builder, self.i)
        if self.s is not None:
            AttributeProtoAddS(builder, s)
        if self.t is not None:
            AttributeProtoAddT(builder, t)
        if self.g is not None:
            AttributeProtoAddG(builder, g)
        if self.sparseTensor is not None:
            AttributeProtoAddSparseTensor(builder, sparseTensor)
        if self.tp is not None:
            AttributeProtoAddTp(builder, tp)
        if self.floats is not None:
            AttributeProtoAddFloats(builder, floats)
        if self.ints is not None:
            AttributeProtoAddInts(builder, ints)
        if self.strings is not None:
            AttributeProtoAddStrings(builder, strings)
        if self.tensors is not None:
            AttributeProtoAddTensors(builder, tensors)
        if self.graphs is not None:
            AttributeProtoAddGraphs(builder, graphs)
        if self.sparseTensors is not None:
            AttributeProtoAddSparseTensors(builder, sparseTensors)
        if self.typeProtos is not None:
            AttributeProtoAddTypeProtos(builder, typeProtos)
        attributeProto = AttributeProtoEnd(builder)
        return attributeProto
