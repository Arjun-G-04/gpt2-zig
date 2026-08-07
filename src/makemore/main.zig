const std = @import("std");
const micrograd = @import("micrograd");
const nn = micrograd.nn;
const engine = micrograd.engine;
const Value = engine.Value;
const Tensor = engine.Tensor;

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    var gpa = std.heap.DebugAllocator(.{}){};
    defer _ = gpa.deinit();
    const ga = gpa.allocator();
    var prng = std.Random.DefaultPrng.init(67);
    const random = prng.random();

    var persistentArena = std.heap.ArenaAllocator.init(ga);
    defer persistentArena.deinit();
    const pa = persistentArena.allocator();

    var tempArena = std.heap.ArenaAllocator.init(ga);
    defer tempArena.deinit();
    const ta = tempArena.allocator();

    const content = try std.Io.Dir.cwd().readFileAlloc(io, "names.txt", pa, .limited(5 * 1024 * 1024));

    // inputs and corresponding outputs of the dataset. dot represents start or end of a word. 0 = . & 1-26 = a-z (convenience)
    var xa: std.ArrayList(u8) = .empty;
    var ya: std.ArrayList(u8) = .empty;

    // populate the xa and ya
    var words = std.mem.splitScalar(u8, content, '\n');
    var ct: usize = 0;
    const noOfWords = 5;
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
            const v = try pa.create(Value);
            v.* = .{ .data = if (i == x) 1 else 0 };
            try enc.append(pa, v);
        }
        try xenc.append(pa, enc.items);
    }

    // basic 27 neurons layer
    const neurons = try nn.createNeuronsSlice(pa, random, 27, 27);
    var layer = nn.Layer{ .neurons = neurons };
    var mlp = nn.MLP{ .layers = &[_]*nn.Layer{&layer} };

    // training
    const start = std.Io.Clock.now(.awake, io);
    var minusOne = Value{ .data = -1 };
    const epoch = 0;
    for (1..epoch + 1) |run| {
        var loss = try ta.create(Value);
        loss.* = .{ .data = 0 };

        for (xenc.items, 0..) |inp, index| {
            // pass each input through nn getting prob for all 27 as some float called as logit
            const logits = try mlp.compute(ta, inp);

            // exponentiate it to get "counts" like postive numbers
            var counts: std.ArrayList(*Value) = .empty;
            for (logits) |logit| {
                try counts.append(ta, try logit.exp(ta));
            }

            // normalize to get probabilities; exponentiation + normalization -> softmax
            var sum = try ta.create(Value);
            sum.* = .{ .data = 0 };
            for (counts.items) |i| {
                sum = try sum.add(ta, i);
            }
            var probs: std.ArrayList(*Value) = .empty;
            for (counts.items) |i| {
                try probs.append(ta, try i.div(ta, sum));
            }

            // calculate loss = negative of log of prob
            const actualOutput = ya.items[index];
            const logOfProb = try probs.items[actualOutput].log(ta);
            const negOfLogOfProb = try logOfProb.mul(ta, &minusOne);
            loss = try loss.add(ta, negOfLogOfProb);
        }
        const noOfItems = try ta.create(Value);
        noOfItems.* = Value{ .data = @as(f32, @floatFromInt(xenc.items.len)) };
        loss = try loss.div(ta, noOfItems);

        std.debug.print("loss {d:.0}: {d:.4}\n", .{ run, loss.data });

        try loss.backward(ta);
        mlp.gradientDescent(0.1);
        _ = tempArena.reset(.retain_capacity);
    }
    const duration = start.untilNow(io, .awake);
    std.debug.print("Time elapsed: {} ms\n", .{duration.toMilliseconds()});

    var adata = [_]f32{ 1, 2, 3, 4 };
    var agrad = [_]f32{0} ** 4;
    var a = Tensor{
        .data = adata[0..],
        .grad = agrad[0..],
        .rows = 2,
        .cols = 2,
    };

    var bdata = [_]f32{ 2, 2 };
    var bgrad = [_]f32{0} ** 2;
    var b = Tensor{
        .data = bdata[0..],
        .grad = bgrad[0..],
        .rows = 2,
        .cols = 1,
    };

    const c = try a.matmul(ta, &b);
    try c.backward(ta);

    a.print(.data);
    b.print(.data);
    c.print(.data);

    a.print(.grad);
    b.print(.grad);
    c.print(.grad);
}
