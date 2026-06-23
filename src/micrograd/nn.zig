const std = @import("std");
const engine = @import("engine.zig");
const Value = engine.Value;

pub const Neuron = struct {
    b: ?*Value = null,
    w: ?std.ArrayList(*Value) = null,

    pub fn init(self: *Neuron, size: usize, a: std.mem.Allocator) !void {
        var seed: [32]u8 = undefined;
        std.crypto.random.bytes(&seed);
        var prng = std.Random.DefaultCsprng.init(seed);
        const random = prng.random();

        self.w = std.ArrayList(*Value){};
        for (0..size) |_| {
            const r = try a.create(Value);
            r.* = Value{.data = -1 + 2 * random.float(f32)};
            try self.w.?.append(a, r);
        }
        const b = try a.create(Value);
        b.* = Value{.data = -1 + 2 * random.float(f32)};
        self.b = b;
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
    pub fn compute(self: *Neuron, x: []const *Value, a: std.mem.Allocator) !*Value {
        if (self.w.?.items.len != x.len) {
            return error.SizeMisMatch;
        }

        var o = try a.create(Value);
        o.* = Value{.data = 0};
        o = try o.add(self.b.?, a);
        for (self.w.?.items, x) |w, i| {
            var m = try a.create(Value);
            m = try w.mul(i, a);
            o = try o.add(m, a);
        }
        
        return o;
    }
};

pub fn createNeuronsArray(aSize: usize, nSize: usize, a: std.mem.Allocator) !std.ArrayList(*Neuron) {
    var array = std.ArrayList(*Neuron){};
    for (0..aSize) |_| {
        const p = try a.create(Neuron);
        p.* = Neuron{};
        try p.init(nSize, a);
        try array.append(a, p);
    }
    return array;
}

pub const Layer = struct {
    neurons: std.ArrayList(*Neuron),

    pub fn compute(self: *Layer, x: []*Value, a: std.mem.Allocator) !std.ArrayList(*Value){
        var o = std.ArrayList(*Value){};
        for (self.neurons.items) |neuron| {
            const y = try neuron.compute(x, a);
            try o.append(a, y);
        }
        return o;
    }
};

pub const MLP = struct {
    layers: std.ArrayList(*Layer),

    pub fn compute(self: *MLP, x: []*Value, a: std.mem.Allocator) !std.ArrayList(*Value) {
        var curr = std.ArrayList(*Value){.items = x};
        for (self.layers.items) |layer| {
            curr = try layer.compute(curr.items, a);
        }
        return curr;
    }
};