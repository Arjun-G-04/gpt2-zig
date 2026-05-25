// for now, implementing in scalar. later can do for tensor.

const std = @import("std");

const Operation = enum { Add, Mul, Tanh, Exp, Pow };

fn addBackward(v: *Value) void {
    const children = v.children.?;
    children[0].?.grad += 1 * v.grad;
    children[1].?.grad += 1 * v.grad;
}

fn mulBackward(v: *Value) void {
    const children = v.children.?;
    children[0].?.grad += children[1].?.data * v.grad;
    children[1].?.grad += children[0].?.data * v.grad;
}

fn tanhBackward(v: *Value) void {
    const children = v.children.?;
    children[0].?.grad += (1 - std.math.pow(f32, v.data, 2)) * v.grad;
}

fn expBackward(v: *Value) void {
    const children = v.children.?;
    children[0].?.grad += v.data * v.grad;
}

fn powBackward(v: *Value) void {
    const children = v.children.?;
    const p = v.power.?;
    children[0].?.grad = (p * std.math.pow(f32, children[0].?.data, p-1)) * v.grad;
}

fn build_top(allocator: std.mem.Allocator, curr: *Value, visited: *std.AutoHashMap(*Value, bool), sorted_nodes: *std.ArrayList(*Value)) void {
    if (visited.contains(curr)) return;
    if (curr.children) |children| {
        for (children) |opt_node| {
            if (opt_node) |node| {
                build_top(allocator, node, visited, sorted_nodes);
            }
        }
    }
    _ = sorted_nodes.append(allocator, curr) catch {};
    _ = visited.put(curr, true) catch {};
}

// Note about pointers.
// - pointers can't be null
// - struct can't contain itself -> because that can be infinite and we can't allocate memory
// - const ptr: *f32 means the pointer can't be changed, but the pointed value can be changed
// - const ptr: *const f32 means the pointed value also cannot be changed via the pointer
const Value = struct {
    data: f32,
    grad: f32 = 0,
    children: ?[2]?*Value = null,
    op: ?Operation = null,
    backward_fn: ?*const fn (self: *Value) void = null,
    power: ?f32 = null,

    pub fn backward(self: *Value) void {
        var gpa = std.heap.GeneralPurposeAllocator(.{}){};
        defer _ = gpa.deinit();
        const allocator = gpa.allocator();

        var visited = std.AutoHashMap(*Value, bool).init(allocator);
        var sorted_nodes = std.ArrayList(*Value){};
        defer visited.deinit();
        defer sorted_nodes.deinit(allocator);

        build_top(allocator, self, &visited, &sorted_nodes);
        std.mem.reverse(*Value, sorted_nodes.items);

        for (sorted_nodes.items) |node| {
            const f = node.backward_fn orelse continue;
            f(node);
        }
    }

    // Note about pointers.
    // If we pass the Value object to this function, then set children of return Value using &,
    // it will be an issue because when Value passed to this function, its scoped to this fn.
    // so on return, the values will be destroyed and hence those pointers created here would be
    // useless. That's why we are giving the input arguments directly as pointers itself.
    pub fn add(self: *Value, other: *Value) Value {
        return .{ .data = self.data + other.data, .children = .{ self, other }, .op = Operation.Add, .backward_fn = addBackward };
    }

    pub fn mul(self: *Value, other: *Value) Value {
        return .{ .data = self.data * other.data, .children = .{ self, other }, .op = Operation.Mul, .backward_fn = mulBackward };
    }

    pub fn tanh(self: *Value) Value {
        const data = (std.math.exp(2 * self.data) - 1) / (std.math.exp(2 * self.data) + 1);
        return .{ .data = data, .children = .{ self, null }, .op = Operation.Tanh, .backward_fn = tanhBackward };
    }

    pub fn exp(self: *Value) Value {
        return .{
            .data = std.math.exp(self.data),
            .children = .{self, null},
            .op = Operation.Exp,
            .backward_fn = expBackward
        };
    }

    pub fn pow(self: *Value, power: f32) Value {
        return .{
            .data = std.math.pow(f32, self.data, power),
            .children = .{self},
            .op = Operation.Pow,
            .backward_fn = powBackward
        };
    }

    pub fn sub(self: *Value, other: *Value) Value {
        return add(self, &mul(Value{.data = -1}), other);
    }
};

const Neuron = struct {
    b: ?*Value = null,
    w: ?std.ArrayList(*Value) = null,

    pub fn init(self: *Neuron, size: usize, allocator: std.mem.Allocator) !void {
        var seed: [32]u8 = undefined;
        std.crypto.random.bytes(&seed);
        var prng = std.Random.DefaultCsprng.init(seed);
        const random = prng.random();
        
        self.w = std.ArrayList(*Value){};
        for (0..size) |_| {
            const r = try allocator.create(Value);
            r.* = Value{.data = -1 + 2 * random.float(f32)};
            try self.w.?.append(allocator, r);
        }
        const b = try allocator.create(Value);
        b.* = Value{.data=-1 + 2 * random.float(f32)};
        self.b = b;
    }

    pub fn compute(self: *Neuron, x: *std.ArrayList(*Value), allocator: std.mem.Allocator) !*Value {
        if (self.w.?.items.len != x.items.len) {
            return error.SizeMisMatch;
        }

        const o = try allocator.create(Value);
        o.* = Value{.data = 0};
        o.* = o.add(self.b.?);
        for (self.w.?.items, x.items) |w, i| {
            const m = try allocator.create(Value);
            m.* = w.mul(i);
            o.* = o.add(m);
        }

        return o;
    }
};

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();

    var arena = std.heap.ArenaAllocator.init(gpa.allocator());
    defer _ = arena.deinit();
    const allocator = arena.allocator();

    var n = Neuron{};
    try n.init(3, allocator);

    var x = std.ArrayList(*Value){};
    defer x.deinit(allocator);

    var v1 = Value{.data=1};
    var v2 = Value{.data=2};
    var v3 = Value{.data=3};
    try x.appendSlice(allocator, &[_]*Value{&v1, &v2, &v3});

    const o = try n.compute(&x, allocator);

    std.debug.print("final value = {d:.4}", .{o.data});
}
