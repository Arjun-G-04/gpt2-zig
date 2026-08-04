const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});
    const micrograd = b.createModule(.{
        .root_source_file = b.path("src/micrograd/micrograd.zig"),
        .target = target,
        .optimize = optimize,
    });

    const exe = b.addExecutable(.{
        .name = "makemore",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/makemore/main.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "micrograd", .module = micrograd },
            },
        }),
    });
    b.installArtifact(exe);

    const fmt_step = b.addFmt(.{
        .paths = &.{ "src", "build.zig" },
    });
    exe.step.dependOn(&fmt_step.step);

    const run_cmd = b.addRunArtifact(exe);
    run_cmd.setCwd(b.path("src/makemore"));
    if (b.args) |args| run_cmd.addArgs(args);

    const run_step = b.step("run", "Run makemore");
    run_step.dependOn(&run_cmd.step);
}
