// for now, implementing in scalar. later can do for tensor.

const std = @import("std");
const engine = @import("engine.zig");

// const Neuron = struct {
//     b: ?*Value = null,
//     w: ?std.ArrayList(*Value) = null,

//     pub fn init(self: *Neuron, size: usize, allocator: std.mem.Allocator) !void {
//         var seed: [32]u8 = undefined;
//         std.crypto.random.bytes(&seed);
//         var prng = std.Random.DefaultCsprng.init(seed);
//         const random = prng.random();

//         self.w = std.ArrayList(*Value){};
//         for (0..size) |_| {
//             const r = try allocator.create(Value);
//             r.* = Value{.data = -1 + 2 * random.float(f32)};
//             try self.w.?.append(allocator, r);
//         }
//         const b = try allocator.create(Value);
//         b.* = Value{.data=-1 + 2 * random.float(f32)};
//         self.b = b;
//     }

//     pub fn compute(self: *Neuron, x: *std.ArrayList(*Value), allocator: std.mem.Allocator) !*Value {
//         if (self.w.?.items.len != x.items.len) {
//             return error.SizeMisMatch;
//         }

//         const o = try allocator.create(Value);
//         o.* = Value{.data = 0};
//         o.* = o.add(self.b.?);
//         for (self.w.?.items, x.items) |w, i| {
//             const m = try allocator.create(Value);
//             m.* = w.mul(i);
//             o.* = o.add(m);
//         }
//         return o;
//     }
// };

// const Layer = struct {
//     neurons: std.ArrayList(*Neuron),

//     pub fn compute(self: *Layer, x: *std.ArrayList(*Value), allocator: std.mem.Allocator) !*std.ArrayList(*Value){
//         var o = try allocator.create(std.ArrayList(*Value));
//         o.* = std.ArrayList(*Value){};
//         for (self.neurons.items) |neuron| {
//             const y = try neuron.compute(x, allocator);
//             try o.append(allocator, y);
//         }
//         return o;
//     }
// };

// const MLP = struct {
//     layers: std.ArrayList(*Layer),

//     pub fn compute(self: *MLP, x: *std.ArrayList(*Value), allocator: std.mem.Allocator) !*std.ArrayList(*Value) {
//         var curr = x;
//         for (self.layers.items) |layer| {
//             curr = try layer.compute(curr, allocator);
//         }
//         return curr;
//     }
// };

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();

    var arena = std.heap.ArenaAllocator.init(gpa.allocator());
    defer _ = arena.deinit();
    const allocator = arena.allocator();

    var test1 = engine.Value{ .data = 10 };
    var test2 = engine.Value{ .data = 12 };

    const test3 = try test1.add(&test2, allocator);
    std.debug.print("{d:.4}", .{test3.data});

    // var n1 = Neuron{};
    // var n2 = Neuron{};
    // var n3 = Neuron{};
    // var n4 = Neuron{};
    // var n5 = Neuron{};
    // try n1.init(2, allocator);
    // try n2.init(2, allocator);
    // try n3.init(2, allocator);
    // try n4.init(2, allocator);
    // try n5.init(2, allocator);

    // var l1_neurons = [_]*Neuron{ &n1, &n2 };
    // var l2_neurons = [_]*Neuron{ &n3, &n4 };
    // var l3_neurons = [_]*Neuron{ &n5 };
    // var l1 = Layer{.neurons = std.ArrayList(*Neuron){.items = l1_neurons[0..]}};
    // var l2 = Layer{.neurons = std.ArrayList(*Neuron){.items = l2_neurons[0..]}};
    // var l3 = Layer{.neurons = std.ArrayList(*Neuron){.items = l3_neurons[0..]}};

    // var mlp_layers = [_]*Layer{ &l1, &l2, &l3 };
    // var mlp = MLP{.layers = std.ArrayList(*Layer){.items = mlp_layers[0..]}};

    // var x1 = Value{.data = 3};
    // var x2 = Value{.data = 4};
    // var o1 = Value{.data = 5};

    // var x3 = Value{.data = 1};
    // var x4 = Value{.data = 2};
    // var o2 = Value{.data = 3};

    // var x5 = Value{.data = 6};
    // var x6 = Value{.data = 7};
    // var o3 = Value{.data = 13};

    // var in1 = [_]*Value{ &x1, &x2 }; var in1_array = std.ArrayList(*Value){.items = in1[0..]};
    // var in2 = [_]*Value{ &x3, &x4 }; var in2_array = std.ArrayList(*Value){.items = in2[0..]};
    // var in3 = [_]*Value{ &x5, &x6 }; var in3_array = std.ArrayList(*Value){.items = in3[0..]};

    // var a1 = try mlp.compute(&in1_array, allocator);
    // var a2 = try mlp.compute(&in2_array, allocator);
    // var a3 = try mlp.compute(&in3_array, allocator);

    // var minus1 = Value{.data = -1};
    // var negative_o1 = minus1.mul(&o1);
    // var inter_1 = a1.items[0].add(&negative_o1);
    // var inter_2 = inter_1.pow(2);

    // var negative_o2 = minus1.mul(&o2);
    // var inter_3 = a2.items[0].add(&negative_o2);
    // var inter_4 = inter_3.pow(2);

    // var negative_o3 = minus1.mul(&o3);
    // var inter_5 = a3.items[0].add(&negative_o3);
    // var inter_6 = inter_5.pow(2);

    // var inter_7 = inter_2.add(&inter_4);
    // var loss = inter_7.add(&inter_6);

    // std.debug.print("LOSS = {d:.4}", .{loss.data});
    // loss.grad = 1;
    // loss.backward();

    // var neurons = [_]*Neuron{&n1, &n2, &n3, &n4, &n5};
    // const all_neurons = std.ArrayList(*Neuron){.items = neurons[0..]};
    // for (all_neurons.items) |neuron| {
    //     for (neuron.w.?.items) |weight| {
    //         weight.data -= 0.01 * weight.grad;
    //     }
    //     neuron.b.?.data -= 0.01 * neuron.b.?.data;
    // }

    // var a12 = try mlp.compute(&in1_array, allocator);
    // var a22 = try mlp.compute(&in2_array, allocator);
    // var a32 = try mlp.compute(&in3_array, allocator);

    // var inter_12 = a12.items[0].sub(&o1);
    // var inter_22 = inter_12.pow(2);

    // var inter_32 = a22.items[0].sub(&o2);
    // var inter_42 = inter_32.pow(2);

    // var inter_52 = a32.items[0].sub(&o3);
    // var inter_62 = inter_52.pow(2);

    // var inter_72 = inter_22.add(&inter_42);
    // const loss2 = inter_72.add(&inter_62);

    // std.debug.print("LOSS = {d:.4}", .{loss2.data});
}
