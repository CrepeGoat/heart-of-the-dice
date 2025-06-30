const std = @import("std");

test "SequenceWithOffset.addDistr" {
    const SeqWOffset = SequenceWithOffset(isize, i32);
    const allocator = std.testing.allocator;

    var buffer = [_]i32{
        1, 2, 3, 4, // array1
        5, 6, 7, // array2
    };
    const seq1 = SeqWOffset{ .index_first = 1, .seq = buffer[0..4] };
    const seq2 = SeqWOffset{ .index_first = 0, .seq = buffer[4..] };

    const result = try seq1.addDistr(allocator, seq2);
    defer result.deinit(allocator);

    try std.testing.expectEqual(1, result.index_first);
    try std.testing.expectEqualSlices(i32, &[_]i32{ 5, 16, 34, 52, 45, 28 }, result.seq);
}

test "SequenceWithOffset.addValues" {
    const SeqWOffset = SequenceWithOffset(isize, i32);
    const allocator = std.testing.allocator;

    var buffer = [_]i32{
        1, 2, 3, 4, // array1
        5, 6, 7, // array2
    };
    const seq1 = SeqWOffset{ .index_first = 1, .seq = buffer[0..4] };
    const seq2 = SeqWOffset{ .index_first = 0, .seq = buffer[4..] };

    const result = try seq1.addValues(allocator, seq2);
    defer result.deinit(allocator);

    try std.testing.expectEqual(0, result.index_first);
    try std.testing.expectEqualSlices(i32, &[_]i32{ 5, 7, 9, 3, 4 }, result.seq);
}

test "SequenceWithOffset.biasBy" {
    const SeqWOffset = SequenceWithOffset(isize, i32);
    const allocator = std.testing.allocator;

    var buffer = [_]i32{ 1, 2, 3, 4 };
    const seq = SeqWOffset{ .index_first = 1, .seq = &buffer };

    const result = try seq.biasBy(allocator, 3);
    defer result.deinit(allocator);

    try std.testing.expectEqual(4, result.index_first);
    try std.testing.expectEqualSlices(i32, seq.seq, result.seq);
}

test "SequenceWithOffset.applyFnToValues" {
    const SeqWOffset = SequenceWithOffset(isize, i32);
    const allocator = std.testing.allocator;

    var buffer = [_]i32{ 1, 2, 3, 4 };
    const seq = SeqWOffset{ .index_first = 1, .seq = &buffer };
    const mapFn = struct {
        pub fn mul2add3(x: i32) i64 {
            return @intCast(2 * x + 3);
        }
    }.mul2add3;

    const result = try seq.applyFnToValues(allocator, mapFn);
    defer result.deinit(allocator);

    try std.testing.expectEqual(seq.index_first, result.index_first);
    try std.testing.expectEqualSlices(i64, &[_]i64{ 5, 7, 9, 11 }, result.seq);
}

test "SequenceWithOffset.scaleBy" {
    const SeqWOffset = SequenceWithOffset(isize, i32);
    const allocator = std.testing.allocator;

    var buffer = [_]i32{ 1, 2, 3, 4 };
    const seq = SeqWOffset{ .index_first = 1, .seq = &buffer };

    const result = try seq.scaleBy(allocator, 5);
    defer result.deinit(allocator);

    try std.testing.expectEqual(seq.index_first, result.index_first);
    try std.testing.expectEqualSlices(i32, &[_]i32{ 5, 10, 15, 20 }, result.seq);
}

/// A sequence of numbers, offset from zero by a set amount.
///
/// The offset avoids having to manually offset data by explicitly storing zeros
/// in the arrays, and also allows arrays to start before zero.
pub fn SequenceWithOffset(X: type, Y: type) type {
    switch (@typeInfo(X)) {
        .Int => {},
        else => unreachable,
    }

    return struct {
        seq: []Y,
        index_first: X,

        const Self = @This();

        fn index_last(self: Self) X {
            return @as(X, @intCast(self.seq.len)) + self.index_first;
        }

        fn copy(self: Self, allocator: std.mem.Allocator) std.mem.Allocator.Error!Self {
            var buffer = try allocator.alloc(Y, self.seq.len);
            @memcpy(buffer[0..], self.seq);

            return .{
                .index_first = self.index_first,
                .seq = buffer,
            };
        }

        pub fn deinit(self: Self, allocator: std.mem.Allocator) void {
            allocator.free(self.seq);
        }

        pub fn addDistr(
            self: Self,
            allocator: std.mem.Allocator,
            other: Self,
        ) std.mem.Allocator.Error!Self {
            return .{
                .index_first = self.index_first + other.index_first,
                .seq = convolve1d(
                    Y,
                    try allocator.alloc(Y, self.seq.len + other.seq.len - 1),
                    self.seq,
                    other.seq,
                ),
            };
        }

        pub fn addValues(
            self: Self,
            allocator: std.mem.Allocator,
            other: Self,
        ) std.mem.Allocator.Error!Self {
            if (self.seq.len == 0) {
                return other.copy(allocator);
            }
            if (other.seq.len == 0) {
                return self.copy(allocator);
            }

            const index_low = @min(self.index_first, other.index_first);
            const index_high = @max(self.index_last(), other.index_last());
            var seq = try allocator.alloc(Y, @intCast(index_high - index_low));

            @memset(seq, 0);
            @memcpy(seq[@as(usize, @intCast(self.index_first - index_low))..], self.seq);
            for (other.seq, @as(usize, @intCast(other.index_first - index_low))..) |si, i| {
                seq[i] += si;
            }

            return .{ .index_first = index_low, .seq = seq };
        }

        pub fn biasBy(self: Self, allocator: std.mem.Allocator, bias: X) std.mem.Allocator.Error!Self {
            var result = try self.copy(allocator);
            result.index_first += bias;
            return result;
        }

        pub fn scaleBy(self: Self, allocator: std.mem.Allocator, scale: Y) std.mem.Allocator.Error!Self {
            var result = try self.copy(allocator);
            for (0..result.seq.len) |i| {
                result.seq[i] *= scale;
            }
            return result;
        }

        pub fn applyFnToValues(
            self: Self,
            allocator: std.mem.Allocator,
            mapFn: anytype,
        ) std.mem.Allocator.Error!SequenceWithOffset(X, @TypeOf(mapFn(self.seq[0]))) {
            // const YNew = comptime switch (@typeInfo(@TypeOf(mapFn))) {
            //     .Fn => |info| info.type orelse unreachable,
            //     else => unreachable,
            // };
            var buffer = try allocator.alloc(@TypeOf(mapFn(self.seq[0])), self.seq.len);
            for (0..buffer.len) |i| {
                buffer[i] = mapFn(self.seq[i]);
            }
            return .{ .index_first = self.index_first, .seq = buffer };
        }
    };
}

test convolve1d {
    const a1 = [_]i32{ 1, 2, 3 };
    const a2 = [_]i32{ 5, 7 };

    var result: [4]i32 = undefined;
    const calc_result = convolve1d(i32, &result, &a1, &a2);

    try std.testing.expectEqual(result[0..], calc_result[0..]);
    try std.testing.expectEqualSlices(i32, &[_]i32{ 5, 17, 29, 21 }, calc_result);
}

fn convolve1d(comptime T: type, result: []T, a1: []const T, a2: []const T) []T {
    const convolve_len = a1.len + a2.len - 1;
    if (result.len != convolve_len) {
        unreachable;
    }
    var _result = result[0..convolve_len];

    for (_result, 0..) |_, i| {
        const a1sub = a1[std.math.sub(usize, i + 1, a2.len) catch 0 .. @min(a1.len, i + 1)];
        const a2sub = a2[std.math.sub(usize, i + 1, a1.len) catch 0 .. @min(a2.len, i + 1)];
        std.debug.assert(a1sub.len == a2sub.len);

        _result[i] = 0;
        for (0..a1sub.len) |j| {
            result[i] += a1sub[j] * a2sub[a1sub.len - j - 1];
        }
    }

    return result[0..convolve_len];
}
