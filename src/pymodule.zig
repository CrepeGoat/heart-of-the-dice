// https://docs.python.org/3/extending/newtypes_tutorial.html

const py = @cImport({
    @cDefine("PY_SSIZE_T_CLEAN", {});
    @cInclude("Python.h");
});

const pycompat = @cImport({
    @cInclude("pycompat.h");
});

const std = @import("std");
const distr = @import("distr");

pub export fn PyInit_distr() [*]py.PyObject {
    return py.PyModuleDef_Init(&distrmodule);
}

const distrmodule = py.PyModuleDef{
    .m_base = py.PyModuleDef_HEAD_INIT,
    .m_name = "distr",
    .m_doc = "A module for manipulating discrete distributions.",
    .m_size = 0,
    .m_slots = distr_module_slots,
};

const distr_module_slots = [_:.{ .slot = 0, .value = null }]py.PyModuleDef_Slot{
    .{
        .slot = py.Py_mod_exec,
        .value = distr_module_exec,
    },
    .{
        .slot = py.Py_mod_multiple_interpreters,
        .value = py.Py_MOD_MULTIPLE_INTERPRETERS_NOT_SUPPORTED,
    },
    // TODO
};

export fn distr_module_exec(m: [*c]py.PyObject) c_int {
    // TODO
    if (py.PyType_Ready(&PyDistributionType) < 0) {
        return -1;
    }

    if (py.PyModule_AddObjectRef(m, "Distribution", @as([*c]py.PyObject, @ptrCast(&PyDistributionType))) < 0) {
        return -1;
    }

    return 0;
}

const PyDistributionType = py.PyTypeObject{
    .ob_base = pycompat.PyVarObject_HEAD_INIT_NULL_0,
    .tp_name = "distr.Distribution",
    .tp_doc = py.PyDoc_STR("Distribution object"),
    .tp_basicsize = @sizeOf(PyDistributionObject),
    .tp_itemsize = @sizeOf(Count),
    .tp_flags = py.Py_TPFLAGS_DEFAULT,
    .tp_new = py.PyType_GenericNew,
};

export const PyDistributionObject = struct {
    obj_base: py.PyVarObject,
    distr: distr.SequenceWithOffset(Offset, Count),
};

// All functions in this namespace should be exposed to the Python interpreter.
// const Api = struct {
//     const Seq = distr.SequenceWithOffset(usize, u64);

//     /// Convert a Distribution to its raw value sequence and offset.
//     fn toRaw(
//         self: [*c]py.PyObject,
//         args: [*c]py.PyObject,
//     ) callconv(.C) ?[*]py.PyObject {
//         var raw_seq: ?[*c]anyopaque = undefined;
//         if (py.PyArg_Parse(args, "?", &raw_seq) == 0) return null;

//         const seq = pyObjToSeq(Offset, Count, raw_seq, allocator);
//     }

//     /// Convert a Distribution to its equivalent probability sequence and offset.
//     fn toProbs(
//         self: [*c]py.PyObject,
//         args: [*c]py.PyObject,
//     ) callconv(.C) ?[*]py.PyObject {}

//     /// Create a Distribution for "rolling no dice".
//     fn roll0(self: [*c]py.PyObject) callconv(.C) ?[*]py.PyObject {}

//     /// Create a Distribution for rolling one die with n sides.
//     fn roll1dn(
//         self: [*c]py.PyObject,
//         args: [*c]py.PyObject,
//     ) callconv(.C) ?[*]py.PyObject {}

//     /// Create a Distribution for repeating a distribution k times, and summing
//     /// the result.
//     fn rollKTimes(
//         self: [*c]py.PyObject,
//         args: [*c]py.PyObject,
//     ) callconv(.C) ?[*]py.PyObject {}

//     /// Create a Distribution for repeating a distribution k + d times, omitting
//     /// the lowest d values, and summing the remaining result.
//     fn rollKTimesDropLow(
//         self: [*c]py.PyObject,
//         args: [*c]py.PyObject,
//     ) callconv(.C) ?[*]py.PyObject {}

//     /// Create a Distribution for repeating a distribution k + d times, omitting
//     /// the highest d values, and summing the remaining result.
//     fn rollKTimesDropHigh(
//         self: [*c]py.PyObject,
//         args: [*c]py.PyObject,
//     ) callconv(.C) ?[*]py.PyObject {}
// };

// fn bytesToSeq(
//     comptime Offset: type,
//     comptime Count: type,
//     pyobj: [*]py.PyObject,
//     allocator: std.mem.Allocator,
// ) distr.SequenceWithOffset(Offset, Count) {}

// fn seqToBytes(
//     comptime Offset: type,
//     comptime Count: type,
//     seq: distr.SequenceWithOffset(Offset, Count),
//     allocator: std.mem.Allocator,
// ) callconv(.C) ?[*]py.PyObject {}

const Offset = usize;
const Count = u64;
const allocator = std.heap.c_allocator;
