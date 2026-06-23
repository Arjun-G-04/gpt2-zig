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

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();

    var arena = std.heap.ArenaAllocator.init(gpa.allocator());
    defer _ = arena.deinit();
    const a = arena.allocator();

    // layers
    var l1 = Layer{.neurons = try createNeuronsArray(2, 2, a)};
    var l2 = Layer{.neurons = try createNeuronsArray(2, 2, a)};
    var l3 = Layer{.neurons = try createNeuronsArray(1, 2, a)};
    var layers = [_]*Layer{&l1, &l2, &l3};

    // mlp
    var mlp = MLP{.layers = std.ArrayList(*Layer){.items = layers[0..]}};
    
    // input
    const x = try createValuesArray(&[_]f32{1,2}, a);

    // output
    const finalValues = try mlp.compute(x.items, a);
    std.debug.print("final value: {d:.4}", .{finalValues.items[0].data});
}
