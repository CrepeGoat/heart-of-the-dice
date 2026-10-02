// https://codeberg.org/mwaddoups/zig-python-base
const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const python_path: ?std.Build.LazyPath = b.option(
        std.Build.LazyPath,
        "python-path",
        "The path to the target Python interpreter (used to programmatically determine the Python.h include path)",
    );
    const pyhpath: ?std.Build.LazyPath = b.option(
        std.Build.LazyPath,
        "pyhpath",
        "The path from which Python.h should be included",
    );

    const pythonh_include_path: std.Build.LazyPath = blk: {
        if (python_path == null and pyhpath == null) {
            @panic("path for Python.h unknown; must provide precisely one of {`pyhpath`, `python_path`}");
        } else if (pyhpath == null) {
            // TODO
            break :blk .{ .cwd_relative = "/usr/local/include" };
        } else if (python_path == null) {
            break :blk pyhpath.?;
        } else {
            @panic("flags {`pyhpath`, `python_path`} are mutually exclusive; provide only one of them");
        }
    };
    const python_h = b.addTranslateC(.{
        .root_source_file = b.path("src/py.h"),
        .target = target,
        .optimize = optimize,
    });
    python_h.addIncludePath(pythonh_include_path);

    const mod = b.addModule(
        "pydistr",
        .{
            .root_source_file = b.path("src/distr.zig"),
            .target = target,
            .optimize = optimize,
            .single_threaded = true,
            .imports = &.{
                .{ .name = "py", .module = python_h.createModule() },
            },
        },
    );

    const lib = b.addLibrary(.{
        .name = "pymodule_distr",
        .linkage = .dynamic,
        // .version = .{ .major = 0, .minor = 1, .patch = 0 },
        .root_module = mod,
    });

    b.installArtifact(lib);
}
