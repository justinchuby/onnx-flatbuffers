# automatically generated by the FlatBuffers compiler, do not modify

# namespace: onnx

import flatbuffers
from flatbuffers.compat import import_numpy
np = import_numpy()

class GraphProto(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAs(cls, buf, offset=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = GraphProto()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def GetRootAsGraphProto(cls, buf, offset=0):
        """This method is deprecated. Please switch to GetRootAs."""
        return cls.GetRootAs(buf, offset)
    # GraphProto
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # GraphProto
    def Node(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            x = self._tab.Vector(o)
            x += flatbuffers.number_types.UOffsetTFlags.py_type(j) * 4
            x = self._tab.Indirect(x)
            from onnx.NodeProto import NodeProto
            obj = NodeProto()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

    # GraphProto
    def NodeLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # GraphProto
    def NodeIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        return o == 0

    # GraphProto
    def Name(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.String(o + self._tab.Pos)
        return None

    # GraphProto
    def Initializer(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            x = self._tab.Vector(o)
            x += flatbuffers.number_types.UOffsetTFlags.py_type(j) * 4
            x = self._tab.Indirect(x)
            from onnx.TensorProto import TensorProto
            obj = TensorProto()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

    # GraphProto
    def InitializerLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # GraphProto
    def InitializerIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        return o == 0

    # GraphProto
    def SparseInitializer(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(10))
        if o != 0:
            x = self._tab.Vector(o)
            x += flatbuffers.number_types.UOffsetTFlags.py_type(j) * 4
            x = self._tab.Indirect(x)
            from onnx.SparseTensorProto import SparseTensorProto
            obj = SparseTensorProto()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

    # GraphProto
    def SparseInitializerLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(10))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # GraphProto
    def SparseInitializerIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(10))
        return o == 0

    # GraphProto
    def DocString(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(12))
        if o != 0:
            return self._tab.String(o + self._tab.Pos)
        return None

    # GraphProto
    def Input(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(14))
        if o != 0:
            x = self._tab.Vector(o)
            x += flatbuffers.number_types.UOffsetTFlags.py_type(j) * 4
            x = self._tab.Indirect(x)
            from onnx.ValueInfoProto import ValueInfoProto
            obj = ValueInfoProto()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

    # GraphProto
    def InputLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(14))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # GraphProto
    def InputIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(14))
        return o == 0

    # GraphProto
    def Output(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(16))
        if o != 0:
            x = self._tab.Vector(o)
            x += flatbuffers.number_types.UOffsetTFlags.py_type(j) * 4
            x = self._tab.Indirect(x)
            from onnx.ValueInfoProto import ValueInfoProto
            obj = ValueInfoProto()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

    # GraphProto
    def OutputLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(16))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # GraphProto
    def OutputIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(16))
        return o == 0

    # GraphProto
    def ValueInfo(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(18))
        if o != 0:
            x = self._tab.Vector(o)
            x += flatbuffers.number_types.UOffsetTFlags.py_type(j) * 4
            x = self._tab.Indirect(x)
            from onnx.ValueInfoProto import ValueInfoProto
            obj = ValueInfoProto()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

    # GraphProto
    def ValueInfoLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(18))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # GraphProto
    def ValueInfoIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(18))
        return o == 0

    # GraphProto
    def QuantizationAnnotation(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(20))
        if o != 0:
            x = self._tab.Vector(o)
            x += flatbuffers.number_types.UOffsetTFlags.py_type(j) * 4
            x = self._tab.Indirect(x)
            from onnx.TensorAnnotation import TensorAnnotation
            obj = TensorAnnotation()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

    # GraphProto
    def QuantizationAnnotationLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(20))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # GraphProto
    def QuantizationAnnotationIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(20))
        return o == 0

def GraphProtoStart(builder):
    builder.StartObject(9)

def Start(builder):
    GraphProtoStart(builder)

def GraphProtoAddNode(builder, node):
    builder.PrependUOffsetTRelativeSlot(0, flatbuffers.number_types.UOffsetTFlags.py_type(node), 0)

def AddNode(builder, node):
    GraphProtoAddNode(builder, node)

def GraphProtoStartNodeVector(builder, numElems):
    return builder.StartVector(4, numElems, 4)

def StartNodeVector(builder, numElems: int) -> int:
    return GraphProtoStartNodeVector(builder, numElems)

def GraphProtoAddName(builder, name):
    builder.PrependUOffsetTRelativeSlot(1, flatbuffers.number_types.UOffsetTFlags.py_type(name), 0)

def AddName(builder, name):
    GraphProtoAddName(builder, name)

def GraphProtoAddInitializer(builder, initializer):
    builder.PrependUOffsetTRelativeSlot(2, flatbuffers.number_types.UOffsetTFlags.py_type(initializer), 0)

def AddInitializer(builder, initializer):
    GraphProtoAddInitializer(builder, initializer)

def GraphProtoStartInitializerVector(builder, numElems):
    return builder.StartVector(4, numElems, 4)

def StartInitializerVector(builder, numElems: int) -> int:
    return GraphProtoStartInitializerVector(builder, numElems)

def GraphProtoAddSparseInitializer(builder, sparseInitializer):
    builder.PrependUOffsetTRelativeSlot(3, flatbuffers.number_types.UOffsetTFlags.py_type(sparseInitializer), 0)

def AddSparseInitializer(builder, sparseInitializer):
    GraphProtoAddSparseInitializer(builder, sparseInitializer)

def GraphProtoStartSparseInitializerVector(builder, numElems):
    return builder.StartVector(4, numElems, 4)

def StartSparseInitializerVector(builder, numElems: int) -> int:
    return GraphProtoStartSparseInitializerVector(builder, numElems)

def GraphProtoAddDocString(builder, docString):
    builder.PrependUOffsetTRelativeSlot(4, flatbuffers.number_types.UOffsetTFlags.py_type(docString), 0)

def AddDocString(builder, docString):
    GraphProtoAddDocString(builder, docString)

def GraphProtoAddInput(builder, input):
    builder.PrependUOffsetTRelativeSlot(5, flatbuffers.number_types.UOffsetTFlags.py_type(input), 0)

def AddInput(builder, input):
    GraphProtoAddInput(builder, input)

def GraphProtoStartInputVector(builder, numElems):
    return builder.StartVector(4, numElems, 4)

def StartInputVector(builder, numElems: int) -> int:
    return GraphProtoStartInputVector(builder, numElems)

def GraphProtoAddOutput(builder, output):
    builder.PrependUOffsetTRelativeSlot(6, flatbuffers.number_types.UOffsetTFlags.py_type(output), 0)

def AddOutput(builder, output):
    GraphProtoAddOutput(builder, output)

def GraphProtoStartOutputVector(builder, numElems):
    return builder.StartVector(4, numElems, 4)

def StartOutputVector(builder, numElems: int) -> int:
    return GraphProtoStartOutputVector(builder, numElems)

def GraphProtoAddValueInfo(builder, valueInfo):
    builder.PrependUOffsetTRelativeSlot(7, flatbuffers.number_types.UOffsetTFlags.py_type(valueInfo), 0)

def AddValueInfo(builder, valueInfo):
    GraphProtoAddValueInfo(builder, valueInfo)

def GraphProtoStartValueInfoVector(builder, numElems):
    return builder.StartVector(4, numElems, 4)

def StartValueInfoVector(builder, numElems: int) -> int:
    return GraphProtoStartValueInfoVector(builder, numElems)

def GraphProtoAddQuantizationAnnotation(builder, quantizationAnnotation):
    builder.PrependUOffsetTRelativeSlot(8, flatbuffers.number_types.UOffsetTFlags.py_type(quantizationAnnotation), 0)

def AddQuantizationAnnotation(builder, quantizationAnnotation):
    GraphProtoAddQuantizationAnnotation(builder, quantizationAnnotation)

def GraphProtoStartQuantizationAnnotationVector(builder, numElems):
    return builder.StartVector(4, numElems, 4)

def StartQuantizationAnnotationVector(builder, numElems: int) -> int:
    return GraphProtoStartQuantizationAnnotationVector(builder, numElems)

def GraphProtoEnd(builder):
    return builder.EndObject()

def End(builder):
    return GraphProtoEnd(builder)

import onnx.NodeProto
import onnx.SparseTensorProto
import onnx.TensorAnnotation
import onnx.TensorProto
import onnx.ValueInfoProto
try:
    from typing import List
except:
    pass

class GraphProtoT(object):

    # GraphProtoT
    def __init__(self):
        self.node = None  # type: List[onnx.NodeProto.NodeProtoT]
        self.name = None  # type: str
        self.initializer = None  # type: List[onnx.TensorProto.TensorProtoT]
        self.sparseInitializer = None  # type: List[onnx.SparseTensorProto.SparseTensorProtoT]
        self.docString = None  # type: str
        self.input = None  # type: List[onnx.ValueInfoProto.ValueInfoProtoT]
        self.output = None  # type: List[onnx.ValueInfoProto.ValueInfoProtoT]
        self.valueInfo = None  # type: List[onnx.ValueInfoProto.ValueInfoProtoT]
        self.quantizationAnnotation = None  # type: List[onnx.TensorAnnotation.TensorAnnotationT]

    @classmethod
    def InitFromBuf(cls, buf, pos):
        graphProto = GraphProto()
        graphProto.Init(buf, pos)
        return cls.InitFromObj(graphProto)

    @classmethod
    def InitFromPackedBuf(cls, buf, pos=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, pos)
        return cls.InitFromBuf(buf, pos+n)

    @classmethod
    def InitFromObj(cls, graphProto):
        x = GraphProtoT()
        x._UnPack(graphProto)
        return x

    # GraphProtoT
    def _UnPack(self, graphProto):
        if graphProto is None:
            return
        if not graphProto.NodeIsNone():
            self.node = []
            for i in range(graphProto.NodeLength()):
                if graphProto.Node(i) is None:
                    self.node.append(None)
                else:
                    nodeProto_ = onnx.NodeProto.NodeProtoT.InitFromObj(graphProto.Node(i))
                    self.node.append(nodeProto_)
        self.name = graphProto.Name()
        if not graphProto.InitializerIsNone():
            self.initializer = []
            for i in range(graphProto.InitializerLength()):
                if graphProto.Initializer(i) is None:
                    self.initializer.append(None)
                else:
                    tensorProto_ = onnx.TensorProto.TensorProtoT.InitFromObj(graphProto.Initializer(i))
                    self.initializer.append(tensorProto_)
        if not graphProto.SparseInitializerIsNone():
            self.sparseInitializer = []
            for i in range(graphProto.SparseInitializerLength()):
                if graphProto.SparseInitializer(i) is None:
                    self.sparseInitializer.append(None)
                else:
                    sparseTensorProto_ = onnx.SparseTensorProto.SparseTensorProtoT.InitFromObj(graphProto.SparseInitializer(i))
                    self.sparseInitializer.append(sparseTensorProto_)
        self.docString = graphProto.DocString()
        if not graphProto.InputIsNone():
            self.input = []
            for i in range(graphProto.InputLength()):
                if graphProto.Input(i) is None:
                    self.input.append(None)
                else:
                    valueInfoProto_ = onnx.ValueInfoProto.ValueInfoProtoT.InitFromObj(graphProto.Input(i))
                    self.input.append(valueInfoProto_)
        if not graphProto.OutputIsNone():
            self.output = []
            for i in range(graphProto.OutputLength()):
                if graphProto.Output(i) is None:
                    self.output.append(None)
                else:
                    valueInfoProto_ = onnx.ValueInfoProto.ValueInfoProtoT.InitFromObj(graphProto.Output(i))
                    self.output.append(valueInfoProto_)
        if not graphProto.ValueInfoIsNone():
            self.valueInfo = []
            for i in range(graphProto.ValueInfoLength()):
                if graphProto.ValueInfo(i) is None:
                    self.valueInfo.append(None)
                else:
                    valueInfoProto_ = onnx.ValueInfoProto.ValueInfoProtoT.InitFromObj(graphProto.ValueInfo(i))
                    self.valueInfo.append(valueInfoProto_)
        if not graphProto.QuantizationAnnotationIsNone():
            self.quantizationAnnotation = []
            for i in range(graphProto.QuantizationAnnotationLength()):
                if graphProto.QuantizationAnnotation(i) is None:
                    self.quantizationAnnotation.append(None)
                else:
                    tensorAnnotation_ = onnx.TensorAnnotation.TensorAnnotationT.InitFromObj(graphProto.QuantizationAnnotation(i))
                    self.quantizationAnnotation.append(tensorAnnotation_)

    # GraphProtoT
    def Pack(self, builder):
        if self.node is not None:
            nodelist = []
            for i in range(len(self.node)):
                nodelist.append(self.node[i].Pack(builder))
            GraphProtoStartNodeVector(builder, len(self.node))
            for i in reversed(range(len(self.node))):
                builder.PrependUOffsetTRelative(nodelist[i])
            node = builder.EndVector()
        if self.name is not None:
            name = builder.CreateString(self.name)
        if self.initializer is not None:
            initializerlist = []
            for i in range(len(self.initializer)):
                initializerlist.append(self.initializer[i].Pack(builder))
            GraphProtoStartInitializerVector(builder, len(self.initializer))
            for i in reversed(range(len(self.initializer))):
                builder.PrependUOffsetTRelative(initializerlist[i])
            initializer = builder.EndVector()
        if self.sparseInitializer is not None:
            sparseInitializerlist = []
            for i in range(len(self.sparseInitializer)):
                sparseInitializerlist.append(self.sparseInitializer[i].Pack(builder))
            GraphProtoStartSparseInitializerVector(builder, len(self.sparseInitializer))
            for i in reversed(range(len(self.sparseInitializer))):
                builder.PrependUOffsetTRelative(sparseInitializerlist[i])
            sparseInitializer = builder.EndVector()
        if self.docString is not None:
            docString = builder.CreateString(self.docString)
        if self.input is not None:
            inputlist = []
            for i in range(len(self.input)):
                inputlist.append(self.input[i].Pack(builder))
            GraphProtoStartInputVector(builder, len(self.input))
            for i in reversed(range(len(self.input))):
                builder.PrependUOffsetTRelative(inputlist[i])
            input = builder.EndVector()
        if self.output is not None:
            outputlist = []
            for i in range(len(self.output)):
                outputlist.append(self.output[i].Pack(builder))
            GraphProtoStartOutputVector(builder, len(self.output))
            for i in reversed(range(len(self.output))):
                builder.PrependUOffsetTRelative(outputlist[i])
            output = builder.EndVector()
        if self.valueInfo is not None:
            valueInfolist = []
            for i in range(len(self.valueInfo)):
                valueInfolist.append(self.valueInfo[i].Pack(builder))
            GraphProtoStartValueInfoVector(builder, len(self.valueInfo))
            for i in reversed(range(len(self.valueInfo))):
                builder.PrependUOffsetTRelative(valueInfolist[i])
            valueInfo = builder.EndVector()
        if self.quantizationAnnotation is not None:
            quantizationAnnotationlist = []
            for i in range(len(self.quantizationAnnotation)):
                quantizationAnnotationlist.append(self.quantizationAnnotation[i].Pack(builder))
            GraphProtoStartQuantizationAnnotationVector(builder, len(self.quantizationAnnotation))
            for i in reversed(range(len(self.quantizationAnnotation))):
                builder.PrependUOffsetTRelative(quantizationAnnotationlist[i])
            quantizationAnnotation = builder.EndVector()
        GraphProtoStart(builder)
        if self.node is not None:
            GraphProtoAddNode(builder, node)
        if self.name is not None:
            GraphProtoAddName(builder, name)
        if self.initializer is not None:
            GraphProtoAddInitializer(builder, initializer)
        if self.sparseInitializer is not None:
            GraphProtoAddSparseInitializer(builder, sparseInitializer)
        if self.docString is not None:
            GraphProtoAddDocString(builder, docString)
        if self.input is not None:
            GraphProtoAddInput(builder, input)
        if self.output is not None:
            GraphProtoAddOutput(builder, output)
        if self.valueInfo is not None:
            GraphProtoAddValueInfo(builder, valueInfo)
        if self.quantizationAnnotation is not None:
            GraphProtoAddQuantizationAnnotation(builder, quantizationAnnotation)
        graphProto = GraphProtoEnd(builder)
        return graphProto
