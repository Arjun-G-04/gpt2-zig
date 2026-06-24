// for now, implementing in scalar. later can do for tensor.

const std = @import("std");

const engine = @import("engine.zig");
const Value = engine.Value;
const createValuesArray = engine.createValuesArray;

const nn = @import("nn.zig");
const Neuron = nn.Neuron;
const Layer = nn.Layer;
const MLP = nn.MLP;
const createNeuronsArray = nn.createNeuronsArray;
const getLossValue = nn.getLossValue;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();

    var persistentArena = std.heap.ArenaAllocator.init(gpa.allocator());
    defer _ = persistentArena.deinit();
    const pAllocator = persistentArena.allocator();

    var tempArena = std.heap.ArenaAllocator.init(gpa.allocator());
    defer _ = tempArena.deinit();
    const tAllocator = tempArena.allocator();

    // layers
    var l1 = Layer{.neurons = try createNeuronsArray(pAllocator, 2, 2)};
    var l2 = Layer{.neurons = try createNeuronsArray(pAllocator, 2, 2)};
    var l3 = Layer{.neurons = try createNeuronsArray(pAllocator, 2, 2)};
    var l4 = Layer{.neurons = try createNeuronsArray(pAllocator, 1, 2)};
    var layers = [_]*Layer{&l1, &l2, &l3, &l4};

    // mlp
    var mlp = MLP{.layers = std.ArrayList(*Layer){.items = layers[0..]}};

    // ground truth
    // input {1, 2} -> output {1}
    // input {6, 7} -> output {9}

    // Inputs
    const x1 = try createValuesArray(pAllocator, &[_]f32{1,2});
    const x2 = try createValuesArray(pAllocator, &[_]f32{6,7});

    const epoch = 10000;
    for (0..epoch) |i| {
        const o1 = try mlp.compute(tAllocator, x1.items);
        std.debug.print("Run {d} - Input 1: {d:.4}\n", .{i+1, o1.items[0].data});
        const o2 = try mlp.compute(tAllocator, x2.items);
        std.debug.print("Run {d} - Input 2: {d:.4}\n", .{i+1, o2.items[0].data});
        const loss = try getLossValue(tAllocator, &[_]*Value{o1.items[0], o2.items[0]}, &[_]f32{1, 9});
        std.debug.print("Run {d} - Loss: {d:.4}\n", .{i+1, loss.data});
    
        // Backpropagation
        try loss.backward(tAllocator);
    
        // Gradient Descent 
        mlp.gradientDescent(0.0001);

        // Clear all intermediary objects
        _ = tempArena.reset(.retain_capacity);
    }
}
