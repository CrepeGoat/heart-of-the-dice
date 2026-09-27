const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});
    const pyhpath: std.Build.LazyPath = b.option(
        std.Build.LazyPath,
        "pyhpath",
        "The path from which Python.h should be included",
    ) orelse .{ .cwd_relative = "/usr/local/include" };

    const python_h = b.addTranslateC(.{
        .root_source_file = b.path("Python.h"),
        .target = target,
        .optimize = optimize,
    });
    python_h.defineCMacro("PY_SSIZE_T_CLEAN", null);
    python_h.addIncludePath(pyhpath);

    const pycompat_h = b.addTranslateC(.{
        .root_source_file = b.path("src/pycompat.h"),
        .target = target,
        .optimize = optimize,
    });
    _ = pycompat_h;

    const lib_pymodule_distr = b.addLibrary(.{
        .name = "pymodule_distr",
        .linkage = .dynamic,
        .version = .{ .major = 0, .minor = 1, .patch = 0 },
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/pymodule.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });

    b.installArtifact(lib_pymodule_distr);
}
