// for now, implementing in scalar. later can do for tensor.

const std = @import("std");

const Operation = enum {
    Add,
    Mul
};

// Note about pointers.
// - pointers can't be null
// - struct can't contain itself -> because that can be infinite and we can't allocate memory
// - const ptr: *f32 means the pointer can't be changed, but the pointed value can be changed
// - const ptr: *const f32 means the pointed value also cannot be changed via the pointer
const Value = struct {
    data: f32,
    children: ?[2]*const Value = null,
    op: ?Operation = null,

    // Note about pointers.
    // If we pass the Value object to this function, then set children of return Value using &,
    // it will be an issue because when Value passed to this function, its scoped to this fn.
    // so on return, the values will be destroyed and hence those pointers created here would be 
    // useless. That's why we are giving the input arguments directly as pointers itself.
    pub fn add(self: *const Value, other: *const Value) Value {
        return .{ 
            .data = self.data + other.data,
            .children = .{self, other},
            .op = Operation.Add
        };
    }

    pub fn mul(self: *const Value, other: *const Value) Value {
        return .{ 
            .data = self.data * other.data,
            .children = .{self, other},
            .op = Operation.Mul
        };
    }

};

pub fn main() void {
    const a = Value{ .data = 6 };
    const b = Value{ .data = 7 };
    const c = a.add(&b);
    std.debug.print("{d:.4}", .{c.data});
}
