// for now, implementing in scalar. later can do for tensor.

const std = @import("std");

const Operation = enum {
    Add,
    Mul,
    Tanh
};

fn addBackward(v: *Value) void {
    const children = v.children.?;
    children[0].?.grad = 1 * v.grad;
    children[1].?.grad = 1 * v.grad;
}

fn mulBackward(v: *Value) void {
    const children = v.children.?;
    children[0].?.grad = children[1].data * v.grad;
    children[1].?.grad = children[0].data * v.grad;
}

fn tanhBackward(v: *Value) void {
    const children = v.children.?;
    children[0].?.grad = (1 - std.math.pow(f32, v.data, 2)) * v.grad;
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

    pub fn backward(self: *Value) void {
        const f = self.backward_fn orelse return;
        f(self);
    }

    // Note about pointers.
    // If we pass the Value object to this function, then set children of return Value using &,
    // it will be an issue because when Value passed to this function, its scoped to this fn.
    // so on return, the values will be destroyed and hence those pointers created here would be 
    // useless. That's why we are giving the input arguments directly as pointers itself.
    pub fn add(self: *Value, other: *Value) Value {
        return .{
            .data = self.data + other.data,
            .children = .{self, other},
            .op = Operation.Add,
            .backward_fn = addBackward
        };
    }

    pub fn mul(self: *Value, other: *Value) Value {
        return .{ 
            .data = self.data * other.data,
            .children = .{self, other},
            .op = Operation.Mul,
            .backward_fn = mulBackward
        };
    }

    pub fn tanh(self: *Value) Value {
        const data = (std.math.exp(2*self.data) - 1)/(std.math.exp(2*self.data) + 1);
        return .{
            .data = data,
            .children = .{self, null},
            .op = Operation.Tanh,
            .backward_fn = tanhBackward
        };
    }

};

pub fn main() void {
    var a = Value{ .data = 0.3 };
    // var b = Value{ .data = 7 };
    var c = a.tanh();
    c.grad = 0.5;
    c.backward();
    std.debug.print("a grad: {d:.4}, c data: {d:.4}, c grad: {d:.4}", .{a.grad, c.data, c.grad});
}
