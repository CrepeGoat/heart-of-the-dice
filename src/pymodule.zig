// https://docs.python.org/3/extending/newtypes_tutorial.html

const py = @import("py");

const pycompat = @cImport({
    @cInclude("pycompat.h");
});

const std = @import("std");
const distr = @import("distr");

pub export fn PyInit_distr() ?*py.PyObject {
    return py.PyModuleDef_Init(&distrmodule);
}

const distrmodule = py.PyModuleDef{
    .m_base = py.PyVarObject_HEAD_INIT_NULL_0,
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
};

export fn distr_module_exec(m: ?*py.PyObject) c_int {
    if (py.PyType_Ready(&PyDistributionType) < 0) {
        return -1;
    }

    if (py.PyModule_AddObjectRef(m, "Distribution", @as(*py.PyObject, @ptrCast(&PyDistributionType))) < 0) {
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
    // .tp_new = py.PyType_GenericNew,
    .tp_new = DistributionApi.new,
    .tp_dealloc = DistributionApi.dealloc,
    .tp_members = distr_exposed_members,
    .tp_methods = distr_exposed_methods,
};

const distr_exposed_members = [_:.{ .name = null }]py.PyMemberDef{};

const distr_exposed_methods = [_:.{ .name = null }]py.PyMethodDef{
    .{
        .name = "roll1dn",
        .ml_meth = DistributionApi.roll1dn,
        .ml_flags = py.METH_VARARGS | py.METH_STATIC,
        .ml_doc = "Generate the distribution for 1dn",
    },
    .{
        .name = "repeat_drop",
        .ml_meth = DistributionApi.repeat_drop,
        .ml_flags = py.METH_VARARGS,
        .ml_doc = "Sum the results of k + |d| independent repetitions of a distribution, dropping the lowest d / highest (-d) outcomes",
    },
    .{
        .name = "contents",
        .ml_meth = DistributionApi.contents,
        .ml_flags = py.METH_NOARGS,
        .ml_doc = "Get the offset and outcome counts for the distribution",
    },
};

const DistributionApi = struct {
    const Self = @This();

    export fn new(pytype: ?*py.PyTypeObject, args: ?*py.PyObject, kwds: ?*py.PyObject) ?*py.PyObject {
        _ = args;
        _ = kwds;
        const self: *PyDistributionObject = @ptrCast(pytype.*.tp_alloc(pytype, 0) orelse return null);

        self.*.distr = CDO.roll0(allocator) catch {
            py.Py_DECREF(self);
            return null;
        };

        return @ptrCast(self);
    }

    export fn dealloc(op: ?*py.PyObject) void {
        const self: *PyDistributionObject = @ptrCast(op);
        self.*.distr.deinit(allocator);
        py.Py_TYPE(self).tp_free(self);
    }

    export fn roll1dn(op: ?*py.PyObject, args: ?*py.PyObject) ?*py.PyObject {
        std.debug.assert(op == null);

        const n: Len = undefined;
        if (py.PyArg_ParseTuple(args, "K", &n) != 0) {
            return null;
        }

        const result: *py.PyObject = Self.new(&PyDistributionType) orelse return null;
        const distr_tmp = CDO.roll1dn(allocator, n) catch {
            py.Py_DECREF(result);
            return null;
        };
        result.*.distr.deinit(allocator);
        result.*.distr = distr_tmp;
        return result;
    }

    export fn repeat_drop(op: ?*py.PyObject, args: ?*py.PyObject) ?*py.PyObject {
        const self: *PyDistributionObject = @ptrCast(op.?);

        const k: DiceCount = undefined;
        const d: SignedDiceCount = undefined;
        if (py.PyArg_ParseTuple(args, "Hh", &k, &d) != 0) {
            return null;
        }

        const result: *py.PyObject = Self.new(&PyDistributionType) orelse return null;
        const distr_tmp = switch (d) {
            0 => CDO.rollKTimes(allocator, self.*.distr, k),
            1...std.math.maxInt(@TypeOf(d)) => CDO.rollKTimesDropLow(
                allocator,
                self.*.distr,
                k,
                @intCast(d),
            ),
            std.math.minInt(@TypeOf(d))...-1 => CDO.rollKTimesDropHigh(
                allocator,
                self.*.distr,
                k,
                @intCast(@abs(d)),
            ),
        } catch |err| {
            _ = err;
            py.Py_DECREF(result);
            return null;
        };
        result.*.distr.deinit(allocator);
        result.*.distr = distr_tmp;
        return result;
    }

    export fn contents(op: ?*py.PyObject, args: ?*py.PyObject) ?*py.PyObject {
        _ = args;

        const self: *PyDistributionObject = @ptrCast(op orelse unreachable);
        const n = self.*.distr.seq.len;

        const counts: *py.PyObject = py.PyTuple_New(n) orelse return null;
        for (0.., self.*.distr.seq) |i, item| {
            const py_item = py.PyLong_FromUnsignedLongLong(item);
            if (py_item == null) {
                for (0..i) |j| {
                    py.Py_DECREF(py.PyTuple_SET_ITEM(counts, @intCast(j)));
                }
                py.Py_DECREF(counts);
                return null;
            }
            py.PyTuple_SET_ITEM(counts, @intCast(i), py_item);
        }

        const offset: *py.PyObject = py.PyLong_FromLongLong(self.*.distr.offset) orelse {
            for (0..n) |j| {
                py.Py_DECREF(py.PyTuple_SET_ITEM(counts, @intCast(j)));
            }
            py.Py_DECREF(counts);
            return null;
        };

        const result: *py.PyObject = py.PyTuple_Pack(2, offset, counts) orelse {
            for (0..n) |j| {
                py.Py_DECREF(py.PyTuple_SET_ITEM(counts, @intCast(j)));
            }
            py.Py_DECREF(counts);
            py.Py_DECREF(offset);
            return null;
        };

        return result;
    }
};

const PyDistributionObject = extern struct {
    obj_base: py.PyVarObject,
    distr: distr.SequenceWithOffset(Offset, Count),
};

const Offset = c_longlong;
const Len = c_ulonglong;
const Count = c_ulonglong;
const DiceCount = c_ushort;
const SignedDiceCount = c_short;
const CDO = distr.CountDiceOutcomes(DiceCount, Offset, Count);
// https://docs.python.org/3/c-api/memory.html
const allocator = std.heap.c_allocator;
