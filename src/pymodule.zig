// https://csprimer.com/watch/varint-extension/
// inspired by https://github.com/adamserafini/zaml/blob/27b2d54ffb39aace5d5d58f0aa75396c3e6fe84d/zamlmodule.zig

const py = @cImport({
    @cDefine("PY_SSIZE_T_CLEAN", {});
    @cInclude("Python.h");
});

const std = @import("std");
const distr = @import("distr");

var CVarintMethods = [_]py.PyMethodDef{
    .{
        .ml_name = "encode",
        .ml_meth = cvarint_encode,
        .ml_flags = py.METH_VARARGS,
        .ml_doc = "Encode an integer as varint.",
    },
    .{
        .ml_name = "decode",
        .ml_meth = cvarint_decode,
        .ml_flags = py.METH_VARARGS,
        .ml_doc = "Decode varint bytes to an integer.",
    },
    .{
        .ml_name = null,
        .ml_meth = null,
        .ml_flags = 0,
        .ml_doc = null,
    },
};

var cvarintmodule = py.PyModuleDef{
    .m_base = py.PyModuleDef_Base{
        .ob_base = py.PyObject{
            .ob_refcnt = 1,
            .ob_type = null,
        },
        .m_init = null,
        .m_index = 0,
        .m_copy = null,
    },
    .m_name = "distr",
    .m_doc = "A library for manipulating discrete distributions.",
    .m_size = -1,
    .m_methods = &CVarintMethods,
    .m_slots = null,
    .m_traverse = null,
    .m_clear = null,
    .m_free = null,
};

pub export fn PyInit_cvarint() [*]py.PyObject {
    return py.PyModule_Create(&cvarintmodule);
}

const Api = struct {
    const Seq = distr.SequenceWithOffset(usize, u64);

    /// Convert a Distribution to its raw value sequence and offset.
    fn toRaw(
        self: [*c]py.PyObject,
        args: [*c]py.PyObject,
    ) callconv(.C) ?[*]py.PyObject {}

    /// Convert a Distribution to its equivalent probability sequence and offset.
    fn toProbs(
        self: [*c]py.PyObject,
        args: [*c]py.PyObject,
    ) callconv(.C) ?[*]py.PyObject {}

    /// Create a Distribution for "rolling no dice".
    fn roll0(self: [*c]py.PyObject) callconv(.C) ?[*]py.PyObject {}

    /// Create a Distribution for rolling one die with n sides.
    fn roll1dn(
        self: [*c]py.PyObject,
        args: [*c]py.PyObject,
    ) callconv(.C) ?[*]py.PyObject {}

    /// Create a Distribution for repeating a distribution k times, and summing
    /// the result.
    fn rollKTimes(
        self: [*c]py.PyObject,
        args: [*c]py.PyObject,
    ) callconv(.C) ?[*]py.PyObject {}

    /// Create a Distribution for repeating a distribution k + d times, omitting
    /// the lowest d values, and summing the remaining result.
    fn rollKTimesDropLow(
        self: [*c]py.PyObject,
        args: [*c]py.PyObject,
    ) callconv(.C) ?[*]py.PyObject {}

    /// Create a Distribution for repeating a distribution k + d times, omitting
    /// the highest d values, and summing the remaining result.
    fn rollKTimesDropHigh(
        self: [*c]py.PyObject,
        args: [*c]py.PyObject,
    ) callconv(.C) ?[*]py.PyObject {}
};

fn pyObjToSeq(
    comptime Offset: type,
    Count: type,
    pyobj: [*]py.PyObject,
) distr.SequenceWithOffset(Offset, Count) {}

fn seqToPyObj(
    comptime Offset: type,
    Count: type,
    seq: distr.SequenceWithOffset(Offset, Count),
) callconv(.C) ?[*]py.PyObject {}
