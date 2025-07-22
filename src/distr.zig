const std = @import("std");

test "CountDiceOutcomes.roll0" {
    const CDO = CountDiceOutcomes(u32, usize, u64);
    const allocator = std.testing.allocator;

    const result = try CDO.roll0(allocator);
    defer result.deinit(allocator);
    try std.testing.expectEqual(0, result.index_first);
    try std.testing.expectEqualSlices(u64, &[_]u64{1}, result.seq);
}

test "CountDiceOutcomes.roll1dn" {
    const CDO = CountDiceOutcomes(u32, usize, u64);
    const allocator = std.testing.allocator;

    inline for (0..10) |n| {
        const result = try CDO.roll1dn(allocator, @intCast(n));
        defer result.deinit(allocator);

        try std.testing.expectEqual(1, result.index_first);
        try std.testing.expectEqualSlices(u64, &([_]u64{1} ** n), result.seq);
    }
}

test "CountDiceOutcomes - rollkdn" {
    const CDO = CountDiceOutcomes(u32, usize, u64);
    const allocator = std.testing.allocator;

    var k: u32 = 1;
    while (k <= 5) : (k += 1) {
        for (1..10) |n| {
            const r1 = try CDO.roll1dn(allocator, n);
            const result = try CDO.rollKTimes(allocator, r1, k);
            r1.deinit(allocator);
            defer result.deinit(allocator);

            const brute_result = try generate_distr_by_brute_force(allocator, struct {
                pub fn f(v: []const u64) u64 {
                    var sum: u64 = 0;
                    for (v) |vi| {
                        sum += vi + 1; // add one, since sides start at 1 but `NestedRangeIterator` starts at 0
                    }
                    return sum;
                }
            }.f, k, n);
            defer brute_result.deinit(allocator);

            try std.testing.expectEqual(brute_result.index_first, result.index_first);
            try std.testing.expectEqualSlices(u64, brute_result.seq, result.seq);
        }
    }
}

test "CountDiceOutcomes - roll kd6 drop highest d" {
    const CDO = CountDiceOutcomes(u32, usize, u64);
    const allocator = std.testing.allocator;

    const r1 = try CDO.roll1dn(allocator, 6);
    defer r1.deinit(allocator);

    inline for (1..4) |k| {
        inline for (1..3) |d| {
            const result = try CDO.rollKTimesDropHigh(allocator, r1, @intCast(k), @intCast(d));
            defer result.deinit(allocator);

            const brute_result = try generate_distr_by_brute_force(allocator, struct {
                pub fn f(v: []const u64) u64 {
                    var sorted = allocator.alloc(u64, v.len) catch unreachable;
                    defer allocator.free(sorted);
                    @memcpy(sorted, v);
                    std.mem.sort(u64, sorted, {}, std.sort.desc(u64));

                    var sum: u64 = 0;
                    for (sorted[d..]) |vi| {
                        sum += vi + 1; // add one, since sides start at 1 but `NestedRangeIterator` starts at 0
                    }
                    return sum;
                }
            }.f, @intCast(k + d), 6);
            defer brute_result.deinit(allocator);

            try std.testing.expectEqual(brute_result.index_first, result.index_first);
            try std.testing.expectEqualSlices(u64, brute_result.seq, result.seq);
        }
    }
}

fn generate_distr_by_brute_force(
    allocator: std.mem.Allocator,
    mapFn: fn (v: []const u64) u64,
    k: u32,
    n: u64,
) !SequenceWithOffset(usize, u64) {
    var values = std.ArrayList(u64).init(allocator);

    const buffer = try allocator.alloc(u64, k);
    defer allocator.free(buffer);
    var iter = NestedRangeIterator.init(buffer, n - 1);
    while (true) {
        const value = mapFn(iter.get());
        const value_usize = @as(usize, @intCast(value));
        if (value_usize >= values.items.len) {
            try values.appendNTimes(0, 1 + value_usize - values.items.len);
        }
        values.items[value_usize] += 1;

        if (!iter.increment()) break;
    }

    const i = for (0..values.items.len) |i| {
        if (values.items[i] != 0) break i;
    } else values.items.len;
    try values.replaceRange(0, i, &[0]u64{});

    return .{ .index_first = i, .seq = try values.toOwnedSlice() };
}

test NestedRangeIterator {
    const allocator = std.testing.allocator;
    const buffer = try allocator.alloc(u64, 3);
    defer allocator.free(buffer);

    var iter = NestedRangeIterator.init(buffer, 1);

    try std.testing.expectEqualSlices(u64, &[_]u64{ 0, 0, 0 }, iter.get());
    try std.testing.expect(iter.increment());
    try std.testing.expectEqualSlices(u64, &[_]u64{ 0, 0, 1 }, iter.get());
    try std.testing.expect(iter.increment());
    try std.testing.expectEqualSlices(u64, &[_]u64{ 0, 1, 1 }, iter.get());
    try std.testing.expect(iter.increment());
    try std.testing.expectEqualSlices(u64, &[_]u64{ 0, 1, 0 }, iter.get());
    try std.testing.expect(iter.increment());
    try std.testing.expectEqualSlices(u64, &[_]u64{ 1, 1, 0 }, iter.get());
    try std.testing.expect(iter.increment());
    try std.testing.expectEqualSlices(u64, &[_]u64{ 1, 1, 1 }, iter.get());
    try std.testing.expect(iter.increment());
    try std.testing.expectEqualSlices(u64, &[_]u64{ 1, 0, 1 }, iter.get());
    try std.testing.expect(iter.increment());
    try std.testing.expectEqualSlices(u64, &[_]u64{ 1, 0, 0 }, iter.get());
    try std.testing.expect(!iter.increment());
}

const NestedRangeIterator = struct {
    buffer: []u64,
    count: u64,

    const Self = @This();

    pub fn init(buffer: []u64, count: u64) Self {
        @memset(buffer, 0);
        return .{ .buffer = buffer, .count = count };
    }

    pub fn get(self: Self) []const u64 {
        return self.buffer;
    }

    pub fn increment(self: *Self) bool {
        return self.increment_inner(true);
    }

    fn increment_inner(self: *Self, is_positive: bool) bool {
        if (self.buffer.len == 0) {
            return false;
        }
        const end_value = if (is_positive) self.count else 0;
        const next_is_positive = is_positive == (self.buffer[0] % 2 == 0);

        {
            var buffer = self.buffer;
            self.buffer = buffer[1..];
            defer self.buffer = buffer;

            if (self.increment_inner(next_is_positive)) {
                return true;
            }
        }

        if (self.buffer[0] > self.count) {
            unreachable;
        } else if (self.buffer[0] == end_value) {
            return false;
        } else {
            if (is_positive) {
                self.buffer[0] += 1;
            } else {
                self.buffer[0] -= 1;
            }
            return true;
        }
    }
};

pub fn CountDiceOutcomes(D: type, X: type, Y: type) type {
    const SeqWOffset = SequenceWithOffset(X, Y);
    const Result = std.mem.Allocator.Error!SeqWOffset;

    return struct {
        pub fn roll0(allocator: std.mem.Allocator) Result {
            return SeqWOffset.initSingle(allocator, 0, 1);
        }

        pub fn roll1dn(allocator: std.mem.Allocator, n: X) Result {
            const buffer = try allocator.alloc(Y, @intCast(n));
            @memset(buffer, 1);
            return SeqWOffset{ .index_first = 1, .seq = buffer };
        }

        pub fn rollKTimes(
            allocator: std.mem.Allocator,
            roll1: SeqWOffset,
            k: D,
        ) (std.mem.Allocator.Error || error{Overflow})!SeqWOffset {
            var result = try roll0(allocator);
            var result_tmp: SeqWOffset = undefined;

            for (0..@intCast(k)) |_| {
                std.mem.swap(SeqWOffset, &result_tmp, &result);
                result = try result_tmp.addDistr(allocator, roll1);

                result_tmp.deinit(allocator);
                result_tmp = undefined;
            }

            return result;
        }

        pub fn rollKTimesDropHigh(
            allocator: std.mem.Allocator,
            roll1: SeqWOffset,
            keep_count: D,
            drop_count: D,
        ) (std.mem.Allocator.Error || error{Overflow})!SeqWOffset {
            const dice_count: usize = @intCast(keep_count + drop_count);

            var dp1 = try allocator.alloc(SeqWOffset, dice_count + 1);
            defer allocator.free(dp1);
            var dp2 = try allocator.alloc(SeqWOffset, dice_count + 1);
            defer allocator.free(dp2);

            std.debug.assert(dp1.len > 0);
            var i1_alloc: usize = 0;
            defer for (0..i1_alloc) |i| {
                dp1[i].deinit(allocator);
            };

            dp1[0] = try roll0(allocator);
            i1_alloc = 1;
            for (1..dp1.len) |i| {
                dp1[i] = try roll1dn(allocator, 0);
                i1_alloc = i + 1;
            }

            for (1..roll1.seq.len + 1) |n| {
                std.mem.swap(@TypeOf(dp1), &dp1, &dp2);
                i1_alloc = 0;
                defer for (0..dp2.len) |i| {
                    dp2[i].deinit(allocator);
                };

                dp1[0] = try roll0(allocator);
                i1_alloc = 1;

                for (1..dp1.len) |i| {
                    const drop_i = std.math.sub(usize, i, keep_count) catch 0;

                    dp1[i] = try roll1dn(allocator, 0);
                    i1_alloc = i + 1;

                    for (0..i + 1) |j| {
                        var tmp = try dp2[i - j].copy(allocator);
                        defer tmp.deinit(allocator);

                        try tmp.biasBy(try std.math.mul(
                            X,
                            @intCast(std.math.sub(usize, j, drop_i) catch 0),
                            try std.math.add(X, roll1.index_first, @intCast(n - 1)),
                        ));
                        try tmp.scaleBy(
                            try std.math.mul(
                                Y,
                                try binomial(Y, @intCast(i), @intCast(j)),
                                try powi_noUnderflow(Y, roll1.seq[n - 1], @intCast(j)),
                            ),
                        );

                        const result = try dp1[i].addValues(allocator, tmp);
                        dp1[i].deinit(allocator);
                        dp1[i] = result;
                    }
                }
            }

            i1_alloc = dp1.len - 1;
            return dp1[dp1.len - 1];
        }
    };
}

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

    {
        const result = try seq1.addValues(allocator, seq2);
        defer result.deinit(allocator);

        try std.testing.expectEqual(0, result.index_first);
        try std.testing.expectEqualSlices(i32, &[_]i32{ 5, 7, 9, 3, 4 }, result.seq);
    }

    {
        const result = try seq2.addValues(allocator, seq1);
        defer result.deinit(allocator);

        try std.testing.expectEqual(0, result.index_first);
        try std.testing.expectEqualSlices(i32, &[_]i32{ 5, 7, 9, 3, 4 }, result.seq);
    }
}

test "SequenceWithOffset.biasBy" {
    const SeqWOffset = SequenceWithOffset(isize, i32);

    var buffer = [_]i32{ 1, 2, 3, 4 };
    var seq = SeqWOffset{ .index_first = 1, .seq = &buffer };
    try seq.biasBy(3);

    try std.testing.expectEqual(4, seq.index_first);
    try std.testing.expectEqualSlices(i32, &[_]i32{ 1, 2, 3, 4 }, seq.seq);
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

    var buffer = [_]i32{ 1, 2, 3, 4 };
    var seq = SeqWOffset{ .index_first = 1, .seq = &buffer };
    try seq.scaleBy(5);

    try std.testing.expectEqualSlices(i32, &[_]i32{ 5, 10, 15, 20 }, seq.seq);
}

/// A sequence of numbers, offset from zero by a set amount.
///
/// The offset avoids having to manually offset data by explicitly storing zeros
/// in the arrays, and also allows arrays to start before zero.
pub fn SequenceWithOffset(X: type, Y: type) type {
    switch (@typeInfo(X)) {
        .int => {},
        else => unreachable,
    }

    return struct {
        seq: []Y,
        index_first: X,

        const Self = @This();
        const Result = std.mem.Allocator.Error!Self;

        fn index_last(self: Self) X {
            return @as(X, @intCast(self.seq.len)) + self.index_first;
        }

        fn copy(self: Self, allocator: std.mem.Allocator) Result {
            var buffer = try allocator.alloc(Y, self.seq.len);
            @memcpy(buffer[0..], self.seq);

            return .{
                .index_first = self.index_first,
                .seq = buffer,
            };
        }

        pub fn initSingle(allocator: std.mem.Allocator, pos: X, value: Y) Result {
            const buffer = try allocator.alloc(Y, 1);
            @memset(buffer, value);
            return Self{ .index_first = pos, .seq = buffer };
        }

        pub fn deinit(self: Self, allocator: std.mem.Allocator) void {
            allocator.free(self.seq);
        }

        pub fn addDistr(
            self: Self,
            allocator: std.mem.Allocator,
            other: Self,
        ) (std.mem.Allocator.Error || error{Overflow})!Self {
            return .{
                .index_first = self.index_first + other.index_first,
                .seq = try convolve1d(
                    Y,
                    try allocator.alloc(Y, self.seq.len + other.seq.len - 1),
                    self.seq,
                    other.seq,
                ),
            };
        }

        pub fn addValues(self: Self, allocator: std.mem.Allocator, other: Self) Result {
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
            {
                const index1 = @as(usize, @intCast(self.index_first - index_low));
                const index2 = @as(usize, @intCast(self.index_last() - index_low));
                @memcpy(seq[index1..index2], self.seq);
            }
            {
                const index1 = @as(usize, @intCast(other.index_first - index_low));
                for (other.seq, index1..) |si, i| {
                    seq[i] += si;
                }
            }

            return .{ .index_first = index_low, .seq = seq };
        }

        pub fn biasBy(self: *Self, bias: X) (error{Overflow}!void) {
            self.index_first = try std.math.add(X, self.index_first, bias);
        }

        pub fn scaleBy(self: *Self, scale: Y) (error{Overflow}!void) {
            for (0..self.seq.len) |i| {
                self.seq[i] = try std.math.mul(Y, self.seq[i], scale);
            }
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
    const calc_result = try convolve1d(i32, &result, &a1, &a2);

    try std.testing.expectEqual(result[0..], calc_result[0..]);
    try std.testing.expectEqualSlices(i32, &[_]i32{ 5, 17, 29, 21 }, calc_result);
}

fn convolve1d(comptime T: type, result: []T, a1: []const T, a2: []const T) (error{Overflow}![]T) {
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
            result[i] = try std.math.add(
                T,
                result[i],
                try std.math.mul(T, a1sub[j], a2sub[a1sub.len - j - 1]),
            );
        }
    }

    return result[0..convolve_len];
}

fn binomial(T: type, n: T, k: T) !T {
    if (k > n - k) {
        return binomial(T, n, n - k);
    }
    var result: T = 1;
    for (0..k) |ki| {
        result = try std.math.mul(T, result, n - ki);
        result = std.math.divExact(T, result, ki + 1) catch unreachable;
    }
    return result;
}

fn powi_noUnderflow(comptime T: type, x: T, y: T) (error{Overflow}!T) {
    std.debug.assert(y >= 0);
    return std.math.powi(T, x, y) catch |err| switch (err) {
        error.Underflow => unreachable,
        error.Overflow => return error.Overflow,
    };
}
