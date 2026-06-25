const std = @import("std");
const engine = @import("engine.zig");
const Value = engine.Value;

pub const Neuron = struct {
    b: *Value,
    w: []*Value,

    pub fn init(a: std.mem.Allocator, size: usize, random: std.Random) !*Neuron {
        // Set random weights
        var w = std.ArrayList(*Value){};
        for (0..size) |_| {
            const r = try a.create(Value);
            r.* = Value{.data = -1 + 2 * random.float(f32)};
            try w.append(a, r);
        }

        // Set random bias
        const b = try a.create(Value);
        b.* = Value{.data = -1 + 2 * random.float(f32)};

        // Create the Neuron in heap (persistent)
        const n = try a.create(Neuron);
        n.* = Neuron{.b = b, .w = w.items};
        return n;
    }

    // i) Rather than passing the ArrayList or passing pointer to ArrayList,
    // its better to just pass the slice of *Value itself to this function.
    // Because that's the only thing required here, sending the ArrayList
    // i.e. the metadata the contains the items is just a tiny overhead
    // compared to sending the slice directly.
    // 
    // ii) []x can be cast to []const x. But []const x can't be 
    // cast to []x. Since compute is read only and creates a new Value "o"
    // and we are not modifying x in any way, it makes sense to define it
    // as []const *Value although []*Value also will work.
    pub fn compute(self: *Neuron, a: std.mem.Allocator, x: []const *Value) !*Value {
        if (self.w.len != x.len) {
            return error.SizeMisMatch;
        }

        // bias + dot product of weights and input
        var o = self.b;
        for (self.w, x) |w, i| {
            o = try o.add(a, try w.mul(a, i));
        }
        
        return o;
    }
};

pub fn createNeuronsSlice(a: std.mem.Allocator, random: std.Random, aSize: usize, nSize: usize) ![]*Neuron {
    var array = std.ArrayList(*Neuron){};
    for (0..aSize) |_| {
        const p = try Neuron.init(a, nSize, random);
        try array.append(a, p);
    }
    return array.items;
}

pub const Layer = struct {
    neurons: []*Neuron,

    pub fn compute(self: *Layer, a: std.mem.Allocator,  x: []const *Value) ![]*Value {
        var o = std.ArrayList(*Value){};
        for (self.neurons) |neuron| {
            const y = try neuron.compute(a, x);
            try o.append(a, y);
        }
        return o.items;
    }
};

pub const MLP = struct {
    layers: []const *Layer,

    pub fn compute(self: *MLP, a: std.mem.Allocator, x: []const *Value) ![]const *Value {
        var curr = x;
        for (self.layers) |layer| {
            curr = try layer.compute(a, curr);
        }
        return curr;
    }

    pub fn gradientDescent(self: *MLP, step: f32) void {
        for (self.layers) |layer| {
            for (layer.neurons) |neuron| {
                const bias = neuron.b;
                bias.data -= step * bias.grad;
                bias.grad = 0;
                for (neuron.w) |weight| {
                    weight.data -= step * weight.grad;
                    weight.grad = 0;
                }
            }
        }
    }
};

pub fn getLossValue(a:std.mem.Allocator, output: []const *Value, truth: []const f32) !*Value {
    var l = try a.create(Value);
    l.* = Value{.data = 0};
    for (output, truth) |o, t| {
        const tp = try a.create(Value);
        tp.* = Value{.data = t};
        l = try l.add(a, try (try o.sub(a, tp)).pow(a, 2));
    }
    return l;
}