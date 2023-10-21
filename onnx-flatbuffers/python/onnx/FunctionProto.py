# automatically generated by the FlatBuffers compiler, do not modify

# namespace: onnx

import flatbuffers
from flatbuffers.compat import import_numpy
np = import_numpy()

class FunctionProto(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAs(cls, buf, offset=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = FunctionProto()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def GetRootAsFunctionProto(cls, buf, offset=0):
        """This method is deprecated. Please switch to GetRootAs."""
        return cls.GetRootAs(buf, offset)
    # FunctionProto
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # FunctionProto
    def Name(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.String(o + self._tab.Pos)
        return None

    # FunctionProto
    def Input(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            a = self._tab.Vector(o)
            return self._tab.String(a + flatbuffers.number_types.UOffsetTFlags.py_type(j * 4))
        return ""

    # FunctionProto
    def InputLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # FunctionProto
    def InputIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        return o == 0

    # FunctionProto
    def Output(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            a = self._tab.Vector(o)
            return self._tab.String(a + flatbuffers.number_types.UOffsetTFlags.py_type(j * 4))
        return ""

    # FunctionProto
    def OutputLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # FunctionProto
    def OutputIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        return o == 0

    # FunctionProto
    def Attribute(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(10))
        if o != 0:
            a = self._tab.Vector(o)
            return self._tab.String(a + flatbuffers.number_types.UOffsetTFlags.py_type(j * 4))
        return ""

    # FunctionProto
    def AttributeLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(10))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # FunctionProto
    def AttributeIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(10))
        return o == 0

    # FunctionProto
    def AttributeProto(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(12))
        if o != 0:
            x = self._tab.Vector(o)
            x += flatbuffers.number_types.UOffsetTFlags.py_type(j) * 4
            x = self._tab.Indirect(x)
            from onnx.AttributeProto import AttributeProto
            obj = AttributeProto()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

    # FunctionProto
    def AttributeProtoLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(12))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # FunctionProto
    def AttributeProtoIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(12))
        return o == 0

    # FunctionProto
    def Node(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(14))
        if o != 0:
            x = self._tab.Vector(o)
            x += flatbuffers.number_types.UOffsetTFlags.py_type(j) * 4
            x = self._tab.Indirect(x)
            from onnx.NodeProto import NodeProto
            obj = NodeProto()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

    # FunctionProto
    def NodeLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(14))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # FunctionProto
    def NodeIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(14))
        return o == 0

    # FunctionProto
    def DocString(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(16))
        if o != 0:
            return self._tab.String(o + self._tab.Pos)
        return None

    # FunctionProto
    def OpsetImport(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(18))
        if o != 0:
            x = self._tab.Vector(o)
            x += flatbuffers.number_types.UOffsetTFlags.py_type(j) * 4
            x = self._tab.Indirect(x)
            from onnx.OperatorSetIdProto import OperatorSetIdProto
            obj = OperatorSetIdProto()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

    # FunctionProto
    def OpsetImportLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(18))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # FunctionProto
    def OpsetImportIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(18))
        return o == 0

    # FunctionProto
    def Domain(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(20))
        if o != 0:
            return self._tab.String(o + self._tab.Pos)
        return None

def FunctionProtoStart(builder):
    builder.StartObject(9)

def Start(builder):
    FunctionProtoStart(builder)

def FunctionProtoAddName(builder, name):
    builder.PrependUOffsetTRelativeSlot(0, flatbuffers.number_types.UOffsetTFlags.py_type(name), 0)

def AddName(builder, name):
    FunctionProtoAddName(builder, name)

def FunctionProtoAddInput(builder, input):
    builder.PrependUOffsetTRelativeSlot(1, flatbuffers.number_types.UOffsetTFlags.py_type(input), 0)

def AddInput(builder, input):
    FunctionProtoAddInput(builder, input)

def FunctionProtoStartInputVector(builder, numElems):
    return builder.StartVector(4, numElems, 4)

def StartInputVector(builder, numElems: int) -> int:
    return FunctionProtoStartInputVector(builder, numElems)

def FunctionProtoAddOutput(builder, output):
    builder.PrependUOffsetTRelativeSlot(2, flatbuffers.number_types.UOffsetTFlags.py_type(output), 0)

def AddOutput(builder, output):
    FunctionProtoAddOutput(builder, output)

def FunctionProtoStartOutputVector(builder, numElems):
    return builder.StartVector(4, numElems, 4)

def StartOutputVector(builder, numElems: int) -> int:
    return FunctionProtoStartOutputVector(builder, numElems)

def FunctionProtoAddAttribute(builder, attribute):
    builder.PrependUOffsetTRelativeSlot(3, flatbuffers.number_types.UOffsetTFlags.py_type(attribute), 0)

def AddAttribute(builder, attribute):
    FunctionProtoAddAttribute(builder, attribute)

def FunctionProtoStartAttributeVector(builder, numElems):
    return builder.StartVector(4, numElems, 4)

def StartAttributeVector(builder, numElems: int) -> int:
    return FunctionProtoStartAttributeVector(builder, numElems)

def FunctionProtoAddAttributeProto(builder, attributeProto):
    builder.PrependUOffsetTRelativeSlot(4, flatbuffers.number_types.UOffsetTFlags.py_type(attributeProto), 0)

def AddAttributeProto(builder, attributeProto):
    FunctionProtoAddAttributeProto(builder, attributeProto)

def FunctionProtoStartAttributeProtoVector(builder, numElems):
    return builder.StartVector(4, numElems, 4)

def StartAttributeProtoVector(builder, numElems: int) -> int:
    return FunctionProtoStartAttributeProtoVector(builder, numElems)

def FunctionProtoAddNode(builder, node):
    builder.PrependUOffsetTRelativeSlot(5, flatbuffers.number_types.UOffsetTFlags.py_type(node), 0)

def AddNode(builder, node):
    FunctionProtoAddNode(builder, node)

def FunctionProtoStartNodeVector(builder, numElems):
    return builder.StartVector(4, numElems, 4)

def StartNodeVector(builder, numElems: int) -> int:
    return FunctionProtoStartNodeVector(builder, numElems)

def FunctionProtoAddDocString(builder, docString):
    builder.PrependUOffsetTRelativeSlot(6, flatbuffers.number_types.UOffsetTFlags.py_type(docString), 0)

def AddDocString(builder, docString):
    FunctionProtoAddDocString(builder, docString)

def FunctionProtoAddOpsetImport(builder, opsetImport):
    builder.PrependUOffsetTRelativeSlot(7, flatbuffers.number_types.UOffsetTFlags.py_type(opsetImport), 0)

def AddOpsetImport(builder, opsetImport):
    FunctionProtoAddOpsetImport(builder, opsetImport)

def FunctionProtoStartOpsetImportVector(builder, numElems):
    return builder.StartVector(4, numElems, 4)

def StartOpsetImportVector(builder, numElems: int) -> int:
    return FunctionProtoStartOpsetImportVector(builder, numElems)

def FunctionProtoAddDomain(builder, domain):
    builder.PrependUOffsetTRelativeSlot(8, flatbuffers.number_types.UOffsetTFlags.py_type(domain), 0)

def AddDomain(builder, domain):
    FunctionProtoAddDomain(builder, domain)

def FunctionProtoEnd(builder):
    return builder.EndObject()

def End(builder):
    return FunctionProtoEnd(builder)

import onnx.AttributeProto
import onnx.NodeProto
import onnx.OperatorSetIdProto
try:
    from typing import List
except:
    pass

class FunctionProtoT(object):

    # FunctionProtoT
    def __init__(self):
        self.name = None  # type: str
        self.input = None  # type: List[str]
        self.output = None  # type: List[str]
        self.attribute = None  # type: List[str]
        self.attributeProto = None  # type: List[onnx.AttributeProto.AttributeProtoT]
        self.node = None  # type: List[onnx.NodeProto.NodeProtoT]
        self.docString = None  # type: str
        self.opsetImport = None  # type: List[onnx.OperatorSetIdProto.OperatorSetIdProtoT]
        self.domain = None  # type: str

    @classmethod
    def InitFromBuf(cls, buf, pos):
        functionProto = FunctionProto()
        functionProto.Init(buf, pos)
        return cls.InitFromObj(functionProto)

    @classmethod
    def InitFromPackedBuf(cls, buf, pos=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, pos)
        return cls.InitFromBuf(buf, pos+n)

    @classmethod
    def InitFromObj(cls, functionProto):
        x = FunctionProtoT()
        x._UnPack(functionProto)
        return x

    # FunctionProtoT
    def _UnPack(self, functionProto):
        if functionProto is None:
            return
        self.name = functionProto.Name()
        if not functionProto.InputIsNone():
            self.input = []
            for i in range(functionProto.InputLength()):
                self.input.append(functionProto.Input(i))
        if not functionProto.OutputIsNone():
            self.output = []
            for i in range(functionProto.OutputLength()):
                self.output.append(functionProto.Output(i))
        if not functionProto.AttributeIsNone():
            self.attribute = []
            for i in range(functionProto.AttributeLength()):
                self.attribute.append(functionProto.Attribute(i))
        if not functionProto.AttributeProtoIsNone():
            self.attributeProto = []
            for i in range(functionProto.AttributeProtoLength()):
                if functionProto.AttributeProto(i) is None:
                    self.attributeProto.append(None)
                else:
                    attributeProto_ = onnx.AttributeProto.AttributeProtoT.InitFromObj(functionProto.AttributeProto(i))
                    self.attributeProto.append(attributeProto_)
        if not functionProto.NodeIsNone():
            self.node = []
            for i in range(functionProto.NodeLength()):
                if functionProto.Node(i) is None:
                    self.node.append(None)
                else:
                    nodeProto_ = onnx.NodeProto.NodeProtoT.InitFromObj(functionProto.Node(i))
                    self.node.append(nodeProto_)
        self.docString = functionProto.DocString()
        if not functionProto.OpsetImportIsNone():
            self.opsetImport = []
            for i in range(functionProto.OpsetImportLength()):
                if functionProto.OpsetImport(i) is None:
                    self.opsetImport.append(None)
                else:
                    operatorSetIdProto_ = onnx.OperatorSetIdProto.OperatorSetIdProtoT.InitFromObj(functionProto.OpsetImport(i))
                    self.opsetImport.append(operatorSetIdProto_)
        self.domain = functionProto.Domain()

    # FunctionProtoT
    def Pack(self, builder):
        if self.name is not None:
            name = builder.CreateString(self.name)
        if self.input is not None:
            inputlist = []
            for i in range(len(self.input)):
                inputlist.append(builder.CreateString(self.input[i]))
            FunctionProtoStartInputVector(builder, len(self.input))
            for i in reversed(range(len(self.input))):
                builder.PrependUOffsetTRelative(inputlist[i])
            input = builder.EndVector()
        if self.output is not None:
            outputlist = []
            for i in range(len(self.output)):
                outputlist.append(builder.CreateString(self.output[i]))
            FunctionProtoStartOutputVector(builder, len(self.output))
            for i in reversed(range(len(self.output))):
                builder.PrependUOffsetTRelative(outputlist[i])
            output = builder.EndVector()
        if self.attribute is not None:
            attributelist = []
            for i in range(len(self.attribute)):
                attributelist.append(builder.CreateString(self.attribute[i]))
            FunctionProtoStartAttributeVector(builder, len(self.attribute))
            for i in reversed(range(len(self.attribute))):
                builder.PrependUOffsetTRelative(attributelist[i])
            attribute = builder.EndVector()
        if self.attributeProto is not None:
            attributeProtolist = []
            for i in range(len(self.attributeProto)):
                attributeProtolist.append(self.attributeProto[i].Pack(builder))
            FunctionProtoStartAttributeProtoVector(builder, len(self.attributeProto))
            for i in reversed(range(len(self.attributeProto))):
                builder.PrependUOffsetTRelative(attributeProtolist[i])
            attributeProto = builder.EndVector()
        if self.node is not None:
            nodelist = []
            for i in range(len(self.node)):
                nodelist.append(self.node[i].Pack(builder))
            FunctionProtoStartNodeVector(builder, len(self.node))
            for i in reversed(range(len(self.node))):
                builder.PrependUOffsetTRelative(nodelist[i])
            node = builder.EndVector()
        if self.docString is not None:
            docString = builder.CreateString(self.docString)
        if self.opsetImport is not None:
            opsetImportlist = []
            for i in range(len(self.opsetImport)):
                opsetImportlist.append(self.opsetImport[i].Pack(builder))
            FunctionProtoStartOpsetImportVector(builder, len(self.opsetImport))
            for i in reversed(range(len(self.opsetImport))):
                builder.PrependUOffsetTRelative(opsetImportlist[i])
            opsetImport = builder.EndVector()
        if self.domain is not None:
            domain = builder.CreateString(self.domain)
        FunctionProtoStart(builder)
        if self.name is not None:
            FunctionProtoAddName(builder, name)
        if self.input is not None:
            FunctionProtoAddInput(builder, input)
        if self.output is not None:
            FunctionProtoAddOutput(builder, output)
        if self.attribute is not None:
            FunctionProtoAddAttribute(builder, attribute)
        if self.attributeProto is not None:
            FunctionProtoAddAttributeProto(builder, attributeProto)
        if self.node is not None:
            FunctionProtoAddNode(builder, node)
        if self.docString is not None:
            FunctionProtoAddDocString(builder, docString)
        if self.opsetImport is not None:
            FunctionProtoAddOpsetImport(builder, opsetImport)
        if self.domain is not None:
            FunctionProtoAddDomain(builder, domain)
        functionProto = FunctionProtoEnd(builder)
        return functionProto
