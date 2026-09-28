// https://codeberg.org/mwaddoups/zig-python-base
const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});
    // const pyhpath: std.Build.LazyPath = b.option(
    //     std.Build.LazyPath,
    //     "pyhpath",
    //     "The path from which Python.h should be included",
    // ) orelse .{ .cwd_relative = "/usr/local/include" };

    const python_h = b.addTranslateC(.{
        .root_source_file = b.path("src/py.h"),
        .target = target,
        .optimize = optimize,
    });
    python_h.addIncludePath(.{ .cwd_relative = "/nix/store/8gg2n5r87vsyy8d78n3s90a9r929c14v-python3-3.13.15/include" });

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
