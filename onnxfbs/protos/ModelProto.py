# automatically generated by the FlatBuffers compiler, do not modify

# namespace: onnx

import flatbuffers
from flatbuffers.compat import import_numpy
np = import_numpy()

class ModelProto(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAs(cls, buf, offset=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = ModelProto()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def GetRootAsModelProto(cls, buf, offset=0):
        """This method is deprecated. Please switch to GetRootAs."""
        return cls.GetRootAs(buf, offset)
    # ModelProto
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # ModelProto
    def IrVersion(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int64Flags, o + self._tab.Pos)
        return 0

    # ModelProto
    def OpsetImport(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            x = self._tab.Vector(o)
            x += flatbuffers.number_types.UOffsetTFlags.py_type(j) * 4
            x = self._tab.Indirect(x)
            from onnxfbs.protos.OperatorSetIdProto import OperatorSetIdProto
            obj = OperatorSetIdProto()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

    # ModelProto
    def OpsetImportLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # ModelProto
    def OpsetImportIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        return o == 0

    # ModelProto
    def ProducerName(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            return self._tab.String(o + self._tab.Pos)
        return None

    # ModelProto
    def ProducerVersion(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(10))
        if o != 0:
            return self._tab.String(o + self._tab.Pos)
        return None

    # ModelProto
    def Domain(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(12))
        if o != 0:
            return self._tab.String(o + self._tab.Pos)
        return None

    # ModelProto
    def ModelVersion(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(14))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int64Flags, o + self._tab.Pos)
        return 0

    # ModelProto
    def DocString(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(16))
        if o != 0:
            return self._tab.String(o + self._tab.Pos)
        return None

    # ModelProto
    def Graph(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(18))
        if o != 0:
            x = self._tab.Indirect(o + self._tab.Pos)
            from onnxfbs.protos.GraphProto import GraphProto
            obj = GraphProto()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

    # ModelProto
    def MetadataProps(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(20))
        if o != 0:
            x = self._tab.Vector(o)
            x += flatbuffers.number_types.UOffsetTFlags.py_type(j) * 4
            x = self._tab.Indirect(x)
            from onnxfbs.protos.StringStringEntryProto import StringStringEntryProto
            obj = StringStringEntryProto()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

    # ModelProto
    def MetadataPropsLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(20))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # ModelProto
    def MetadataPropsIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(20))
        return o == 0

    # ModelProto
    def TrainingInfo(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(22))
        if o != 0:
            x = self._tab.Vector(o)
            x += flatbuffers.number_types.UOffsetTFlags.py_type(j) * 4
            x = self._tab.Indirect(x)
            from onnxfbs.protos.TrainingInfoProto import TrainingInfoProto
            obj = TrainingInfoProto()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

    # ModelProto
    def TrainingInfoLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(22))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # ModelProto
    def TrainingInfoIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(22))
        return o == 0

    # ModelProto
    def Functions(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(24))
        if o != 0:
            x = self._tab.Vector(o)
            x += flatbuffers.number_types.UOffsetTFlags.py_type(j) * 4
            x = self._tab.Indirect(x)
            from onnxfbs.protos.FunctionProto import FunctionProto
            obj = FunctionProto()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

    # ModelProto
    def FunctionsLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(24))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # ModelProto
    def FunctionsIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(24))
        return o == 0

def ModelProtoStart(builder):
    builder.StartObject(11)

def Start(builder):
    ModelProtoStart(builder)

def ModelProtoAddIrVersion(builder, irVersion):
    builder.PrependInt64Slot(0, irVersion, 0)

def AddIrVersion(builder, irVersion):
    ModelProtoAddIrVersion(builder, irVersion)

def ModelProtoAddOpsetImport(builder, opsetImport):
    builder.PrependUOffsetTRelativeSlot(1, flatbuffers.number_types.UOffsetTFlags.py_type(opsetImport), 0)

def AddOpsetImport(builder, opsetImport):
    ModelProtoAddOpsetImport(builder, opsetImport)

def ModelProtoStartOpsetImportVector(builder, numElems):
    return builder.StartVector(4, numElems, 4)

def StartOpsetImportVector(builder, numElems: int) -> int:
    return ModelProtoStartOpsetImportVector(builder, numElems)

def ModelProtoAddProducerName(builder, producerName):
    builder.PrependUOffsetTRelativeSlot(2, flatbuffers.number_types.UOffsetTFlags.py_type(producerName), 0)

def AddProducerName(builder, producerName):
    ModelProtoAddProducerName(builder, producerName)

def ModelProtoAddProducerVersion(builder, producerVersion):
    builder.PrependUOffsetTRelativeSlot(3, flatbuffers.number_types.UOffsetTFlags.py_type(producerVersion), 0)

def AddProducerVersion(builder, producerVersion):
    ModelProtoAddProducerVersion(builder, producerVersion)

def ModelProtoAddDomain(builder, domain):
    builder.PrependUOffsetTRelativeSlot(4, flatbuffers.number_types.UOffsetTFlags.py_type(domain), 0)

def AddDomain(builder, domain):
    ModelProtoAddDomain(builder, domain)

def ModelProtoAddModelVersion(builder, modelVersion):
    builder.PrependInt64Slot(5, modelVersion, 0)

def AddModelVersion(builder, modelVersion):
    ModelProtoAddModelVersion(builder, modelVersion)

def ModelProtoAddDocString(builder, docString):
    builder.PrependUOffsetTRelativeSlot(6, flatbuffers.number_types.UOffsetTFlags.py_type(docString), 0)

def AddDocString(builder, docString):
    ModelProtoAddDocString(builder, docString)

def ModelProtoAddGraph(builder, graph):
    builder.PrependUOffsetTRelativeSlot(7, flatbuffers.number_types.UOffsetTFlags.py_type(graph), 0)

def AddGraph(builder, graph):
    ModelProtoAddGraph(builder, graph)

def ModelProtoAddMetadataProps(builder, metadataProps):
    builder.PrependUOffsetTRelativeSlot(8, flatbuffers.number_types.UOffsetTFlags.py_type(metadataProps), 0)

def AddMetadataProps(builder, metadataProps):
    ModelProtoAddMetadataProps(builder, metadataProps)

def ModelProtoStartMetadataPropsVector(builder, numElems):
    return builder.StartVector(4, numElems, 4)

def StartMetadataPropsVector(builder, numElems: int) -> int:
    return ModelProtoStartMetadataPropsVector(builder, numElems)

def ModelProtoAddTrainingInfo(builder, trainingInfo):
    builder.PrependUOffsetTRelativeSlot(9, flatbuffers.number_types.UOffsetTFlags.py_type(trainingInfo), 0)

def AddTrainingInfo(builder, trainingInfo):
    ModelProtoAddTrainingInfo(builder, trainingInfo)

def ModelProtoStartTrainingInfoVector(builder, numElems):
    return builder.StartVector(4, numElems, 4)

def StartTrainingInfoVector(builder, numElems: int) -> int:
    return ModelProtoStartTrainingInfoVector(builder, numElems)

def ModelProtoAddFunctions(builder, functions):
    builder.PrependUOffsetTRelativeSlot(10, flatbuffers.number_types.UOffsetTFlags.py_type(functions), 0)

def AddFunctions(builder, functions):
    ModelProtoAddFunctions(builder, functions)

def ModelProtoStartFunctionsVector(builder, numElems):
    return builder.StartVector(4, numElems, 4)

def StartFunctionsVector(builder, numElems: int) -> int:
    return ModelProtoStartFunctionsVector(builder, numElems)

def ModelProtoEnd(builder):
    return builder.EndObject()

def End(builder):
    return ModelProtoEnd(builder)

import onnx.FunctionProto
import onnx.GraphProto
import onnx.OperatorSetIdProto
import onnx.StringStringEntryProto
import onnx.TrainingInfoProto
try:
    from typing import List, Optional
except:
    pass

class ModelProtoT(object):

    # ModelProtoT
    def __init__(self):
        self.irVersion = 0  # type: int
        self.opsetImport = None  # type: List[onnx.OperatorSetIdProto.OperatorSetIdProtoT]
        self.producerName = None  # type: str
        self.producerVersion = None  # type: str
        self.domain = None  # type: str
        self.modelVersion = 0  # type: int
        self.docString = None  # type: str
        self.graph = None  # type: Optional[onnx.GraphProto.GraphProtoT]
        self.metadataProps = None  # type: List[onnx.StringStringEntryProto.StringStringEntryProtoT]
        self.trainingInfo = None  # type: List[onnx.TrainingInfoProto.TrainingInfoProtoT]
        self.functions = None  # type: List[onnx.FunctionProto.FunctionProtoT]

    @classmethod
    def InitFromBuf(cls, buf, pos):
        modelProto = ModelProto()
        modelProto.Init(buf, pos)
        return cls.InitFromObj(modelProto)

    @classmethod
    def InitFromPackedBuf(cls, buf, pos=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, pos)
        return cls.InitFromBuf(buf, pos+n)

    @classmethod
    def InitFromObj(cls, modelProto):
        x = ModelProtoT()
        x._UnPack(modelProto)
        return x

    # ModelProtoT
    def _UnPack(self, modelProto):
        if modelProto is None:
            return
        self.irVersion = modelProto.IrVersion()
        if not modelProto.OpsetImportIsNone():
            self.opsetImport = []
            for i in range(modelProto.OpsetImportLength()):
                if modelProto.OpsetImport(i) is None:
                    self.opsetImport.append(None)
                else:
                    operatorSetIdProto_ = onnx.OperatorSetIdProto.OperatorSetIdProtoT.InitFromObj(modelProto.OpsetImport(i))
                    self.opsetImport.append(operatorSetIdProto_)
        self.producerName = modelProto.ProducerName()
        self.producerVersion = modelProto.ProducerVersion()
        self.domain = modelProto.Domain()
        self.modelVersion = modelProto.ModelVersion()
        self.docString = modelProto.DocString()
        if modelProto.Graph() is not None:
            self.graph = onnx.GraphProto.GraphProtoT.InitFromObj(modelProto.Graph())
        if not modelProto.MetadataPropsIsNone():
            self.metadataProps = []
            for i in range(modelProto.MetadataPropsLength()):
                if modelProto.MetadataProps(i) is None:
                    self.metadataProps.append(None)
                else:
                    stringStringEntryProto_ = onnx.StringStringEntryProto.StringStringEntryProtoT.InitFromObj(modelProto.MetadataProps(i))
                    self.metadataProps.append(stringStringEntryProto_)
        if not modelProto.TrainingInfoIsNone():
            self.trainingInfo = []
            for i in range(modelProto.TrainingInfoLength()):
                if modelProto.TrainingInfo(i) is None:
                    self.trainingInfo.append(None)
                else:
                    trainingInfoProto_ = onnx.TrainingInfoProto.TrainingInfoProtoT.InitFromObj(modelProto.TrainingInfo(i))
                    self.trainingInfo.append(trainingInfoProto_)
        if not modelProto.FunctionsIsNone():
            self.functions = []
            for i in range(modelProto.FunctionsLength()):
                if modelProto.Functions(i) is None:
                    self.functions.append(None)
                else:
                    functionProto_ = onnx.FunctionProto.FunctionProtoT.InitFromObj(modelProto.Functions(i))
                    self.functions.append(functionProto_)

    # ModelProtoT
    def Pack(self, builder):
        if self.opsetImport is not None:
            opsetImportlist = []
            for i in range(len(self.opsetImport)):
                opsetImportlist.append(self.opsetImport[i].Pack(builder))
            ModelProtoStartOpsetImportVector(builder, len(self.opsetImport))
            for i in reversed(range(len(self.opsetImport))):
                builder.PrependUOffsetTRelative(opsetImportlist[i])
            opsetImport = builder.EndVector()
        if self.producerName is not None:
            producerName = builder.CreateString(self.producerName)
        if self.producerVersion is not None:
            producerVersion = builder.CreateString(self.producerVersion)
        if self.domain is not None:
            domain = builder.CreateString(self.domain)
        if self.docString is not None:
            docString = builder.CreateString(self.docString)
        if self.graph is not None:
            graph = self.graph.Pack(builder)
        if self.metadataProps is not None:
            metadataPropslist = []
            for i in range(len(self.metadataProps)):
                metadataPropslist.append(self.metadataProps[i].Pack(builder))
            ModelProtoStartMetadataPropsVector(builder, len(self.metadataProps))
            for i in reversed(range(len(self.metadataProps))):
                builder.PrependUOffsetTRelative(metadataPropslist[i])
            metadataProps = builder.EndVector()
        if self.trainingInfo is not None:
            trainingInfolist = []
            for i in range(len(self.trainingInfo)):
                trainingInfolist.append(self.trainingInfo[i].Pack(builder))
            ModelProtoStartTrainingInfoVector(builder, len(self.trainingInfo))
            for i in reversed(range(len(self.trainingInfo))):
                builder.PrependUOffsetTRelative(trainingInfolist[i])
            trainingInfo = builder.EndVector()
        if self.functions is not None:
            functionslist = []
            for i in range(len(self.functions)):
                functionslist.append(self.functions[i].Pack(builder))
            ModelProtoStartFunctionsVector(builder, len(self.functions))
            for i in reversed(range(len(self.functions))):
                builder.PrependUOffsetTRelative(functionslist[i])
            functions = builder.EndVector()
        ModelProtoStart(builder)
        ModelProtoAddIrVersion(builder, self.irVersion)
        if self.opsetImport is not None:
            ModelProtoAddOpsetImport(builder, opsetImport)
        if self.producerName is not None:
            ModelProtoAddProducerName(builder, producerName)
        if self.producerVersion is not None:
            ModelProtoAddProducerVersion(builder, producerVersion)
        if self.domain is not None:
            ModelProtoAddDomain(builder, domain)
        ModelProtoAddModelVersion(builder, self.modelVersion)
        if self.docString is not None:
            ModelProtoAddDocString(builder, docString)
        if self.graph is not None:
            ModelProtoAddGraph(builder, graph)
        if self.metadataProps is not None:
            ModelProtoAddMetadataProps(builder, metadataProps)
        if self.trainingInfo is not None:
            ModelProtoAddTrainingInfo(builder, trainingInfo)
        if self.functions is not None:
            ModelProtoAddFunctions(builder, functions)
        modelProto = ModelProtoEnd(builder)
        return modelProto
