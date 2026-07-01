const std = @import("std");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // load the file into heap memory
    const content = try std.fs.cwd().readFileAlloc(allocator, "names.txt", 5 * 1024 * 1024);
    defer allocator.free(content);

    
    var lines = std.mem.splitScalar(u8, content, '\n');
    var i: usize = 0;
    while (lines.next()) |line| {
        for (line) |char| {
            std.debug.print("char: {d}\n", .{char});
        }
        std.debug.print("line {d}: {s}\n", .{i, line});
        i += 1;
        if (i > 9) {
            break;
        }
    }
}