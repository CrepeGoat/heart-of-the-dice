// https://csprimer.com/watch/varint-extension/
// inspired by https://github.com/adamserafini/zaml/blob/27b2d54ffb39aace5d5d58f0aa75396c3e6fe84d/zamlmodule.zig

const std = @import("std");

const varint = @import("protobuf-varint");

const py = @cImport({
    @cDefine("PY_SSIZE_T_CLEAN", {});
    @cInclude("Python.h");
});

fn cvarint_encode(
    self: [*c]py.PyObject,
    args: [*c]py.PyObject,
) callconv(.C) ?[*]py.PyObject {
    _ = self;

    var cvalue_K: CUInt = undefined;
    if (py.PyArg_ParseTuple(args, "K", &cvalue_K) == 0) return null;
    if (!canIntCast(u64, cvalue_K)) return null;
    const value = @as(u64, @intCast(cvalue_K));

    var buffer: [10]varint.VarintByte = undefined;
    const result_vbytes = varint.encode(value, &buffer);

    // see https://docs.python.org/3/c-api/arg.html#building-values
    return py.Py_BuildValue(
        "y#",
        @as([*c]u8, @ptrCast(result_vbytes.ptr)),
        @as(py.Py_ssize_t, @intCast(result_vbytes.len)),
    );
}

fn cvarint_decode(
    self: [*c]py.PyObject,
    args: [*c]py.PyObject,
) callconv(.C) ?[*]py.PyObject {
    _ = self;

    var cvalue_ptr: [*c]u8 = undefined;
    var cvalue_len: py.Py_ssize_t = undefined;
    if (py.PyArg_ParseTuple(args, "y#", &cvalue_ptr, &cvalue_len) == 0) return null;
    if (cvalue_ptr == null) return null;
    const value: []const varint.VarintByte = @ptrCast(cvalue_ptr[0..@as(usize, @intCast(cvalue_len))]);

    const result = varint.decode(value) catch return null;
    return py.Py_BuildValue("K", @as(CUInt, result));
}

/// A c-native uint at least as big as u64.
const CUInt: type = c_ulonglong;
comptime {
    std.debug.assert(@bitSizeOf(CUInt) >= @bitSizeOf(u64));
}

/// Runtime check that an int value fits in another int type.
fn canIntCast(comptime T: type, value: anytype) bool {
    return @as(@TypeOf(value), @truncate(@as(T, @truncate(value)))) == value;
}

var CVarintMethods = [_]py.PyMethodDef{
    py.PyMethodDef{
        .ml_name = "encode",
        .ml_meth = cvarint_encode,
        .ml_flags = py.METH_VARARGS,
        .ml_doc = "Encode an integer as varint.",
    },
    py.PyMethodDef{
        .ml_name = "decode",
        .ml_meth = cvarint_decode,
        .ml_flags = py.METH_VARARGS,
        .ml_doc = "Decode varint bytes to an integer.",
    },
    py.PyMethodDef{
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
    .m_name = "cvarint",
    .m_doc = "A C implementation of protobuf varint encoding",
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
