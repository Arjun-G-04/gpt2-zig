const std = @import("std");

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
    children[0].?.grad += (p * std.math.pow(f32, children[0].?.data, p - 1)) * v.grad;
}

fn topo_sort(a: std.mem.Allocator, curr: *Value, visited: *std.AutoHashMap(*Value, bool), sorted_nodes: *std.ArrayList(*Value)) !void {
    if (visited.contains(curr)) return;
    if (curr.children) |children| {
        for (children) |opt_node| {
            if (opt_node) |node| {
                try topo_sort(a, node, visited, sorted_nodes);
            }
        }
    }
    try sorted_nodes.append(a, curr);
    try visited.put(curr, true);
}

// Note about pointers.
// - pointers can't be null
// - struct can't contain itself -> because that can be infinite and we can't allocate memory
// - const ptr: *f32 means the pointer can't be changed, but the pointed value can be changed
// - const ptr: *const f32 means the pointed value also cannot be changed via the pointer
pub const Value = struct {
    data: f32,
    grad: f32 = 0,
    children: ?[2]?*Value = null,
    backward_fn: ?*const fn (self: *Value) void = null,
    power: ?f32 = null,

    pub fn backward(self: *Value, a: std.mem.Allocator) !void {
        var visited = std.AutoHashMap(*Value, bool).init(a);
        var sorted_nodes = std.ArrayList(*Value){};
        defer visited.deinit();
        defer sorted_nodes.deinit(a);

        try topo_sort(a, self, &visited, &sorted_nodes);
        std.mem.reverse(*Value, sorted_nodes.items);

        for (sorted_nodes.items) |node| {
            const f = node.backward_fn orelse continue;
            f(node);
        }
    }

    // Note about pointers.
    // If we pass the Value object to this function, then set children of return Value using &,
    // it will be an issue because when Value passed to this function, its scoped to this fn. (stack mem.)
    // so on return, the values will be destroyed and hence those pointers created here would be
    // useless. That's why we are giving the input arguments directly as pointers itself and
    // we are also allocating the new Value in a heap of the main program and returning a pointer
    // to that value in that main persistant heap.
    pub fn add(self: *Value, other: *Value, a: std.mem.Allocator) !*Value {
        // try is used for running a function that returns an Error Union. It means that the function
        // can succeed or it can fail and return an error. try is shorthand for: if the fn succeeds,
        // then return the value, if it fails then propagate the error, as if error made in this line.
        const p = try a.create(Value);
        p.* = .{ .data = self.data + other.data, .children = .{ self, other }, .backward_fn = addBackward };
        return p;
    }

    pub fn mul(self: *Value, other: *Value, a: std.mem.Allocator) !*Value {
        const p = try a.create(Value);
        p.* = .{ .data = self.data * other.data, .children = .{ self, other }, .backward_fn = mulBackward };
        return p;
    }

    pub fn tanh(self: *Value, a: std.mem.Allocator) !*Value {
        const p = try a.create(Value);
        p.* = .{ .data = std.math.tanh(self.data), .children = .{ self, null }, .backward_fn = tanhBackward };
        return p;
    }

    pub fn exp(self: *Value, a: std.mem.Allocator) !*Value {
        const p = try a.create(Value);
        p.* = .{ .data = std.math.exp(self.data), .children = .{ self, null }, .backward_fn = expBackward };
        return p;
    }

    pub fn pow(self: *Value, power: f32, a: std.mem.Allocator) !*Value {
        const p = try a.create(Value);
        p.* = .{
            .data = std.math.pow(f32, self.data, power),
            .children = .{ self, null },
            .backward_fn = powBackward,
            .power = power,
        };
        return p;
    }

    pub fn sub(self: *Value, other: *Value, a: std.mem.Allocator) !*Value {
        const minus_one = try a.create(Value);
        minus_one.* = .{ .data = -1 };
        const negative_other = try minus_one.mul(other, a);
        return self.add(negative_other, a);
    }
};
