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
            v.* = .{.data = if (i == x) 1 else 0};
            try enc.append(pa, v);
        }
        try xenc.append(pa, enc.items);
    }

    // random
    const rng_impl: std.Random.IoSource = .{ .io = io };
    const random = rng_impl.interface();

    // basic 27 neurons layer
    const neurons = try nn.createNeuronsSlice(pa, random, 27, 27);
    var layer = nn.Layer{.neurons = neurons};
    var mlp = nn.MLP{.layers = &[_]*nn.Layer{&layer}};

    // calculate loss
    var loss = try ta.create(Value);
    loss.* = .{.data = 0};
    var minusOne = Value{.data = -1};
    for (xenc.items, 0..) |inp, index| {
        // pass each input through nn getting prob for all 27 as some float - logit
        const logits = try mlp.compute(ta, inp);

        // exponentiate it to get "counts" like
        var counts: std.ArrayList(*Value) = .empty;
        for (logits) |logit| {
            try counts.append(ta, try logit.exp(ta));
        }

        // normalize to get probabilities
        var sum: f32 = 0;
        for (counts.items) |i| {
            sum += i.data;
        }
        const reciprocalOfSum = try ta.create(Value);
        reciprocalOfSum.* = Value{.data = 1 / sum};
        var probs: std.ArrayList(*Value) = .empty;
        for (counts.items) |i| {
            try probs.append(ta, try i.mul(ta, reciprocalOfSum));
        }

        // calculate loss = negative of log of prob
        const actualOutput = ya.items[index];
        const logOfProb = try probs.items[actualOutput].log(ta);
        const negOfLogOfProb = try logOfProb.mul(ta, &minusOne);
        loss = try loss.add(ta, negOfLogOfProb);
    }
    var reciprocalNoOfItems = Value{.data = 1 / @as(f32, @floatFromInt(xenc.items.len))};
    loss = try loss.mul(ta, &reciprocalNoOfItems);

    std.debug.print("loss 1: {d:.4}\n", .{loss.data});

    try loss.backward(ta);
    mlp.gradientDescent(0.001);
    _ = tempArena.reset(.retain_capacity);

    loss = try ta.create(Value);
    loss.* = .{.data = 0};
    for (xenc.items, 0..) |inp, index| {
        // pass each input through nn getting prob for all 27 as some float - logit
        const logits = try mlp.compute(ta, inp);

        // exponentiate it to get "counts" like
        var counts: std.ArrayList(*Value) = .empty;
        for (logits) |logit| {
            try counts.append(ta, try logit.exp(ta));
        }

        // normalize to get probabilities
        var sum: f32 = 0;
        for (counts.items) |i| {
            sum += i.data;
        }
        const reciprocalOfSum = try ta.create(Value);
        reciprocalOfSum.* = Value{.data = 1 / sum};
        var probs: std.ArrayList(*Value) = .empty;
        for (counts.items) |i| {
            try probs.append(ta, try i.mul(ta, reciprocalOfSum));
        }

        // calculate loss = negative of log of prob
        // log of prob = likelihood. since xa:ya pair exist in data set, prob of that should be 1
        // so less than 1 means more loss, close to 1 means less loss. log(prob) gives that (with neg)
        const actualOutput = ya.items[index];
        const logOfProb = try probs.items[actualOutput].log(ta);
        const negOfLogOfProb = try logOfProb.mul(ta, &minusOne);
        loss = try loss.add(ta, negOfLogOfProb);
    }
    loss = try loss.mul(ta, &reciprocalNoOfItems);

    std.debug.print("loss 2: {d:.4}", .{loss.data});
}
