const std = @import("std");

fn addBackward(v: *Value) void {
    v.children[0].?.grad += 1 * v.grad;
    v.children[1].?.grad += 1 * v.grad;
}

fn mulBackward(v: *Value) void {
    v.children[0].?.grad += v.children[1].?.data * v.grad;
    v.children[1].?.grad += v.children[0].?.data * v.grad;
}

fn tanhBackward(v: *Value) void {
    v.children[0].?.grad += (1 - std.math.pow(f32, v.data, 2)) * v.grad;
}

fn expBackward(v: *Value) void {
    v.children[0].?.grad += v.data * v.grad;
}

fn powBackward(v: *Value) void {
    const p = v.power.?;
    v.children[0].?.grad += (p * std.math.pow(f32, v.children[0].?.data, p - 1)) * v.grad;
}

fn topo_sort(a: std.mem.Allocator, curr: *Value, visited: *std.AutoHashMap(*Value, bool), sorted_nodes: *std.ArrayList(*Value)) !void {
    if (visited.contains(curr)) return;
    for (curr.children) |opt_node| {
        if (opt_node) |node| {
            try topo_sort(a, node, visited, sorted_nodes);
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
    children: [2]?*Value = .{null, null},
    backward_fn: ?*const fn (self: *Value) void = null,
    power: ?f32 = null,

    pub fn backward(self: *Value, a: std.mem.Allocator) !void {
        var visited = std.AutoHashMap(*Value, bool).init(a);
        var sorted_nodes = std.ArrayList(*Value){};
        defer visited.deinit();
        defer sorted_nodes.deinit(a);

        try topo_sort(a, self, &visited, &sorted_nodes);
        std.mem.reverse(*Value, sorted_nodes.items);

        // grad of the last node with itself is 1.
        self.grad = 1;
        
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
    pub fn add(self: *Value, a: std.mem.Allocator, other: *Value) !*Value {
        // try is used for running a function that returns an Error Union. It means that the function
        // can succeed or it can fail and return an error. try is shorthand for: if the fn succeeds,
        // then return the value, if it fails then propagate the error, as if error made in this line.
        const p = try a.create(Value);
        p.* = .{ .data = self.data + other.data, .children = .{ self, other }, .backward_fn = addBackward };
        return p;
    }

    pub fn mul(self: *Value, a: std.mem.Allocator, other: *Value) !*Value {
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

    pub fn pow(self: *Value, a: std.mem.Allocator, power: f32) !*Value {
        const p = try a.create(Value);
        p.* = .{
            .data = std.math.pow(f32, self.data, power),
            .children = .{ self, null },
            .backward_fn = powBackward,
            .power = power,
        };
        return p;
    }

    pub fn sub(self: *Value, a: std.mem.Allocator, other: *Value) !*Value {
        const minus_one = try a.create(Value);
        minus_one.* = .{ .data = -1 };
        const negative_other = try minus_one.mul(a, other);
        return self.add(a, negative_other);
    }
};

pub fn createValuesSlice(a: std.mem.Allocator, i: []const f32) ![]*Value {
    // ArrayList itself is just usually 24 bytes of metadata.
    // It stores where the actual slice is located and what is
    // its size. So there isn't a need to create this "metadata"
    // in the heap. Thus, its fine to return it as value and it 
    // will just copy that tiny metadata to the main fn.
    // 
    // Additional Note: Previously I was using ArrayList here.
    // However, as this values "array" is going to be read only 
    // and won't be needed to append or modify after creation
    // we can just return the slice and use the slice itself
    // in the main program.
    var array = std.ArrayList(*Value){};
    for (i) |d| {
        const p = try a.create(Value);
        p.* = Value{.data = d};
        try array.append(a, p);
    }
    return array.items;
}