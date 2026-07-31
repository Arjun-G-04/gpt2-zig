const std = @import("std");
const micrograd = @import("micrograd");
const nn = micrograd.nn;
const engine = micrograd.engine;
const Value = engine.Value;

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    var gpa = std.heap.DebugAllocator(.{}){};
    defer _ = gpa.deinit();
    const ga = gpa.allocator();

    var persistentArena = std.heap.ArenaAllocator.init(ga);
    defer persistentArena.deinit();
    const pa = persistentArena.allocator();

    var tempArena = std.heap.ArenaAllocator.init(ga);
    defer tempArena.deinit();
    const ta = tempArena.allocator();

    // load the file into heap memory
    const content = try std.Io.Dir.cwd().readFileAlloc(io, "names.txt", pa, .limited(5 * 1024 * 1024));

    // inputs and corresponding outputs of the dataset. 0 = . & 1-26 = a-z (convenience)
    var xa: std.ArrayList(u8) = .empty;
    var ya: std.ArrayList(u8) = .empty;

    // populate the xa and ya
    var words = std.mem.splitScalar(u8, content, '\n');
    var ct: usize = 0;
    const noOfWords = 1;
    while (words.next()) |word| {
        try xa.append(pa, 0);
        try ya.append(pa, word[0] - 'a' + 1);
        for (0..word.len - 1) |i| {
            try xa.append(pa, word[i] - 'a' + 1);
            try ya.append(pa, word[i + 1] - 'a' + 1);
        }
        try xa.append(pa, word[word.len - 1] - 'a' + 1);
        try ya.append(pa, 0);

        ct += 1;
        if (ct > noOfWords - 1) break;
    }

    // one-hot encode the input array
    var xenc: std.ArrayList([]*Value) = .empty;
    for (xa.items) |x| {
        var enc: std.ArrayList(*Value) = .empty;
        for (0..27) |i| {
            var v = try pa.create(Value);
            v.data = if (i == x) 1 else 0;
            try enc.append(pa, v);
        }
        try xenc.append(pa, enc.items);
    }

    // random
    const rng_impl: std.Random.IoSource = .{ .io = io };
    const random = rng_impl.interface();

    // pass the one-hot encoded input array to a very simple, one layer nn
    const neurons = try nn.createNeuronsSlice(pa, random, 27, 27);
    var layer = nn.Layer{.neurons = neurons};
    var mlp = nn.MLP{.layers = &[_]*nn.Layer{&layer}};
    const output = try mlp.compute(ta, xenc.items[0]);

    for (output) |i| {
        std.debug.print("{d:.4}\n", .{i.data});
    }
}
