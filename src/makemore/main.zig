const std = @import("std");

pub fn main(init: std.process.Init) !void {
    const io = init.io;

    var gpa = std.heap.DebugAllocator(.{}){};
    defer _ = gpa.deinit();
    const a = gpa.allocator();

    // load the file into heap memory
    const content = try std.Io.Dir.cwd().readFileAlloc(io, "names.txt", a, .limited(5 * 1024 * 1024));
    defer a.free(content);

    // inputs and corresponding outputs of the dataset. 0 = . & 1-26 = a-z (convenience)
    var xa: std.ArrayList(u8) = .empty;
    defer xa.deinit(a);
    var ya: std.ArrayList(u8) = .empty;
    defer ya.deinit(a);

    // populate the xa and ya
    var words = std.mem.splitScalar(u8, content, '\n');
    var c: usize = 0;
    while (words.next()) |word| {
        try xa.append(a, 0);
        try ya.append(a, word[0] - 'a' + 1);
        for (0..word.len - 1) |i| {
            try xa.append(a, word[i] - 'a' + 1);
            try ya.append(a, word[i + 1] - 'a' + 1);
        }
        try xa.append(a, word[word.len - 1] - 'a' + 1);
        try ya.append(a, 0);

        c += 1;
        if (c > 2) break;
    }

    // one-hot encode the input array
    var xenc: std.ArrayList([27]u8) = .empty;
    defer xenc.deinit(a);
    for (xa.items) |x| {
        var s: [27]u8 = .{0} ** 27;
        s[x] = 1;
        try xenc.append(a, s);
    }

    for (xenc.items) |x| {
        for (x) |b| {
            std.debug.print("{d}", .{b});
        }
        std.debug.print("\n", .{});
    }
}
