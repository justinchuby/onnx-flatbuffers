# automatically generated by the FlatBuffers compiler, do not modify

# namespace: onnx

import flatbuffers
from flatbuffers.compat import import_numpy
np = import_numpy()

class TensorAnnotation(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAs(cls, buf, offset=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = TensorAnnotation()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def GetRootAsTensorAnnotation(cls, buf, offset=0):
        """This method is deprecated. Please switch to GetRootAs."""
        return cls.GetRootAs(buf, offset)
    # TensorAnnotation
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # TensorAnnotation
    def TensorName(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.String(o + self._tab.Pos)
        return None

    # TensorAnnotation
    def QuantParameterTensorNames(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            x = self._tab.Vector(o)
            x += flatbuffers.number_types.UOffsetTFlags.py_type(j) * 4
            x = self._tab.Indirect(x)
            from onnx.StringStringEntryProto import StringStringEntryProto
            obj = StringStringEntryProto()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

    # TensorAnnotation
    def QuantParameterTensorNamesLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # TensorAnnotation
    def QuantParameterTensorNamesIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        return o == 0

def TensorAnnotationStart(builder):
    builder.StartObject(2)

def Start(builder):
    TensorAnnotationStart(builder)

def TensorAnnotationAddTensorName(builder, tensorName):
    builder.PrependUOffsetTRelativeSlot(0, flatbuffers.number_types.UOffsetTFlags.py_type(tensorName), 0)

def AddTensorName(builder, tensorName):
    TensorAnnotationAddTensorName(builder, tensorName)

def TensorAnnotationAddQuantParameterTensorNames(builder, quantParameterTensorNames):
    builder.PrependUOffsetTRelativeSlot(1, flatbuffers.number_types.UOffsetTFlags.py_type(quantParameterTensorNames), 0)

def AddQuantParameterTensorNames(builder, quantParameterTensorNames):
    TensorAnnotationAddQuantParameterTensorNames(builder, quantParameterTensorNames)

def TensorAnnotationStartQuantParameterTensorNamesVector(builder, numElems):
    return builder.StartVector(4, numElems, 4)

def StartQuantParameterTensorNamesVector(builder, numElems: int) -> int:
    return TensorAnnotationStartQuantParameterTensorNamesVector(builder, numElems)

def TensorAnnotationEnd(builder):
    return builder.EndObject()

def End(builder):
    return TensorAnnotationEnd(builder)
