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

fn logBackward(v: *Value) void {
    v.children[0].?.grad += v.grad / v.children[0].?.data;
}

fn divBackward(v: *Value) void {
    v.children[0].?.grad += (1 / v.children[1].?.data) * v.grad;
    v.children[1].?.grad += (-v.children[0].?.data / std.math.pow(f32, v.children[1].?.data, 2)) * v.grad;
}

fn getValueChildren(v: *Value) [2]?*Value {
    return v.children;
}

fn getTensorChildren(t: *Tensor) [2]?*Tensor {
    return t.children;
}

fn topo_sort(comptime T: type, a: std.mem.Allocator, curr: *T, getTChildren: fn (*T) [2]?*T, visited: *std.AutoHashMap(*T, bool), sorted_nodes: *std.ArrayList(*T)) !void {
    if (visited.contains(curr)) return;
    try visited.put(curr, true);
    for (getTChildren(curr)) |opt_node| {
        if (opt_node) |node| {
            try topo_sort(T, a, node, getTChildren, visited, sorted_nodes);
        }
    }
    try sorted_nodes.append(a, curr);
}

// Note about pointers.
// - pointers can't be null
// - struct can't contain itself -> because that can be infinite and we can't allocate memory
// - const ptr: *f32 means the pointer can't be changed, but the pointed value can be changed
// - const ptr: *const f32 means the pointed value also cannot be changed via the pointer
pub const Value = struct {
    data: f32,
    grad: f32 = 0,
    children: [2]?*Value = .{ null, null },
    backward_fn: ?*const fn (self: *Value) void = null,
    power: ?f32 = null,

    pub fn backward(self: *Value, a: std.mem.Allocator) !void {
        var visited = std.AutoHashMap(*Value, bool).init(a);
        var sorted_nodes: std.ArrayList(*Value) = .empty;
        defer visited.deinit();
        defer sorted_nodes.deinit(a);

        try topo_sort(Value, a, self, getValueChildren, &visited, &sorted_nodes);
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

    pub fn log(self: *Value, a: std.mem.Allocator) !*Value {
        const p = try a.create(Value);
        p.* = .{ .data = std.math.log(f32, std.math.e, self.data), .children = .{ self, null }, .backward_fn = logBackward };
        return p;
    }

    pub fn div(self: *Value, a: std.mem.Allocator, other: *Value) !*Value {
        const p = try a.create(Value);
        p.* = .{ .data = (self.data / other.data), .children = .{ self, other }, .backward_fn = divBackward };
        return p;
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
        p.* = Value{ .data = d };
        try array.append(a, p);
    }
    return array.items;
}

// lets rock with Tensorsssss (2nd order i.e. 2D for now. nth order tensor later)

fn tensorAddBackward(t: *Tensor) void {
    for (t.grad, 0..) |g, i| {
        t.children[0].?.grad[i] += g;
        t.children[1].?.grad[i] += g;
    }
}

fn tensorRowBroadcastBackward(t: *Tensor) void {
    for (t.grad, 0..) |g, i| {
        t.children[0].?.grad[i] += g;
        t.children[1].?.grad[i % t.children[1].?.cols] += g;
    }
}

fn tensorColBroadcastBackward(t: *Tensor) void {
    for (t.grad, 0..) |g, i| {
        t.children[0].?.grad[i] += g;
        t.children[1].?.grad[i / t.children[1].?.rows] += g;
    }
}

fn matmulBackward(t: *Tensor) void {
    const left = t.children[0].?;
    const right = t.children[1].?;

    for (0..left.rows) |i| {
        for (0..right.cols) |j| {
            for (0..left.cols) |k| {
                left.put(.grad, i, k, left.get(.grad, i, k) + t.get(.grad, i, j) * right.get(.data, k, j));
                right.put(.grad, k, j, right.get(.grad, k, j) + t.get(.grad, i, j) * left.get(.data, i, k));
            }
        }
    }
}

const Item = enum { data, grad };

pub const Tensor = struct {
    data: []f32,
    grad: []f32,
    rows: usize,
    cols: usize,
    children: [2]?*Tensor = .{ null, null },
    backwardFn: ?*const fn (self: *Tensor) void = null,

    pub fn get(self: *Tensor, item: Item, r: usize, c: usize) f32 {
        const items = if (item == .data) self.data else self.grad;
        // index of an element is: row_no * col_size + col_no
        return items[r * self.cols + c];
    }

    pub fn put(self: *Tensor, item: Item, r: usize, c: usize, v: f32) void {
        const items = if (item == .data) self.data else self.grad;
        items[r * self.cols + c] = v;
    }

    pub fn print(self: *Tensor, item: Item) void {
        const items = if (item == .data) self.data else self.grad;
        for (items, 0..) |d, i| {
            std.debug.print("{d:.2} ", .{d});
            if (i % self.cols == self.cols - 1) std.debug.print("\n", .{});
        }
        std.debug.print("\n", .{});
    }

    pub fn backward(self: *Tensor, a: std.mem.Allocator) !void {
        for (0..self.grad.len) |i| self.grad[i] = 1;
        var visited = std.AutoHashMap(*Tensor, bool).init(a);
        var sorted: std.ArrayList(*Tensor) = .empty;
        try topo_sort(Tensor, a, self, getTensorChildren, &visited, &sorted);
        std.mem.reverse(*Tensor, sorted.items);
        for (sorted.items) |node| {
            const f = node.backwardFn orelse continue;
            f(node);
        }
    }

    pub fn add(self: *Tensor, a: std.mem.Allocator, other: *Tensor) !*Tensor {
        const output = try a.create(Tensor);
        var data: std.ArrayList(f32) = .empty;
        var backwardFn: ?*const fn (self: *Tensor) void = null;
        const grad = try a.alloc(f32, self.rows * self.cols);
        @memset(grad, 0);

        if (other.rows == self.rows and other.cols == self.cols) {
            for (self.data, other.data) |x, y| try data.append(a, x + y);
            backwardFn = tensorAddBackward;
        } else if (other.rows == self.rows and other.cols == 1) {
            for (self.data, 0..) |x, i| try data.append(a, x + other.data[i % other.rows]);
            backwardFn = tensorColBroadcastBackward;
        } else if (other.cols == self.cols and other.rows == 1) {
            for (self.data, 0..) |x, i| try data.append(a, x + other.data[i % other.cols]);
            backwardFn = tensorRowBroadcastBackward;
        } else {
            a.destroy(output);
            return error.ShapeMismatch;
        }

        output.* = .{ .data = data.items, .grad = grad, .rows = self.rows, .cols = self.cols, .children = .{ self, other }, .backwardFn = backwardFn };
        return output;
    }

    pub fn matmul(self: *Tensor, a: std.mem.Allocator, other: *Tensor) !*Tensor {
        if (self.cols != other.rows) return error.ShapeMismatch;

        const output = try a.create(Tensor);
        var data: std.ArrayList(f32) = .empty;
        const grad = try a.alloc(f32, self.rows * other.cols);
        @memset(grad, 0);

        // for each row in self
        for (0..self.rows) |i| {
            // for each col in other
            for (0..other.cols) |j| {
                var sum: f32 = 0;
                // taking each item of the current row in self
                for (0..self.cols) |k| {
                    // multiplying the corresponding element in col of other
                    sum += self.get(.data, i, k) * other.get(.data, k, j);
                }
                try data.append(a, sum);
            }
        }

        output.* = .{ .data = data.items, .grad = grad, .rows = self.rows, .cols = other.cols, .children = .{ self, other }, .backwardFn = matmulBackward };
        return output;
    }
};
