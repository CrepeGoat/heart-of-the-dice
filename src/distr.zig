const std = @import("std");

// Using `std.Random` instead of fuzz testing with `std.testing.Smith`
// due to bug:
// https://ziggit.dev/t/errors-when-trying-to-run-std-testing-fuzz-on-0-16/15515
// https://codeberg.org/ziglang/zig/issues/30655
// TODO move back to fuzz testing framework when bug is resolved
fn generateRngIntSeq(
    // smith: *std.testing.Smith,
    rng: std.Random,
    allocator: std.mem.Allocator,
    comptime X: type,
    comptime Y: type,
    offset_min: X,
    offset_max: X,
    len_min: X,
    len_max: X,
    val_min: Y,
    val_max: Y,
) std.mem.Allocator.Error!SequenceWithOffset(X, Y) {
    const len =
        // smith.valueRangeAtMost(X, len_min, len_max);
        rng.intRangeAtMost(X, len_min, len_max);
    const buffer = try allocator.alloc(Y, len);
    for (buffer) |*item| {
        item.* =
            // smith.valueRangeAtMost(Y, val_min, val_max);
            rng.intRangeAtMost(Y, val_min, val_max);
    }
    const startIndex =
        // smith.valueRangeAtMost(X, offset_min, offset_max);
        rng.intRangeAtMost(X, offset_min, offset_max);

    return .{
        .seq = buffer,
        .index_first = startIndex,
    };
}

test "CountDiceOutcomes.roll0" {
    const CDO = CountDiceOutcomes(u32, usize, u64);
    const allocator = std.testing.allocator;

    const result = try CDO.roll0(allocator);
    defer result.deinit(allocator);
    try std.testing.expectEqual(0, result.index_first);
    try std.testing.expectEqualSlices(u64, &[_]u64{1}, result.seq);
}

test "CountDiceOutcomes - roll0 deallocates on error" {
    const CDO = CountDiceOutcomes(u32, usize, u64);
    const allocator = std.testing.allocator;

    const test_fn = struct {
        pub fn func(alloc: std.mem.Allocator) !void {
            const roll1 = try CDO.roll0(alloc);
            defer roll1.deinit(alloc);
        }
    }.func;

    try std.testing.checkAllAllocationFailures(allocator, test_fn, .{});
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

test "CountDiceOutcomes - roll1d10 deallocates on error" {
    const CDO = CountDiceOutcomes(u32, usize, u64);
    const allocator = std.testing.allocator;

    const test_fn = struct {
        pub fn func(alloc: std.mem.Allocator) !void {
            const roll1 = try CDO.roll1dn(alloc, 10);
            defer roll1.deinit(alloc);
        }
    }.func;

    try std.testing.checkAllAllocationFailures(allocator, test_fn, .{});
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

test "CountDiceOutcomes - roll k times - fuzz test deallocations under errors" {
    const CDO = CountDiceOutcomes(u32, usize, u8);

    const test_fn = struct {
        fn func(
            allocator: std.mem.Allocator,
            roll1: SequenceWithOffset(usize, u8),
            k: u32,
        ) !void {
            const result = CDO.rollKTimes(allocator, roll1, k) catch |err| switch (err) {
                error.Overflow => return {},
                error.OutOfMemory => return error.OutOfMemory,
            };
            defer result.deinit(allocator);
        }
    }.func;

    const allocator = std.testing.allocator;
    var prng: std.Random.DefaultPrng = .init(std.testing.random_seed);
    for (0..100) |_| {
        const roll1: SequenceWithOffset(usize, u8) = try generateRngIntSeq(
            prng.random(),
            allocator,
            usize,
            u8,
            0,
            10,
            1,
            20,
            std.math.minInt(u8),
            std.math.maxInt(u8),
        );
        defer roll1.deinit(allocator);

        const k =
            // smith.valueRangeAtMost(u32, 1, 5);
            prng.random().intRangeAtMost(u32, 1, 5);

        try std.testing.checkAllAllocationFailures(allocator, test_fn, .{ roll1, k });
    }
}

test "CountDiceOutcomes - roll6d10 deallocates on error" {
    const CDO = CountDiceOutcomes(u32, usize, u64);
    const allocator = std.testing.allocator;

    const test_fn = struct {
        pub fn func(alloc: std.mem.Allocator) !void {
            const roll1 = try CDO.roll1dn(alloc, 10);
            defer roll1.deinit(alloc);

            const result = try CDO.rollKTimes(alloc, roll1, 6);
            defer result.deinit(alloc);
        }
    }.func;

    try std.testing.checkAllAllocationFailures(allocator, test_fn, .{});
}

test "CountDiceOutcomes - roll kd6 drop lowest d" {
    const CDO = CountDiceOutcomes(u32, usize, u64);
    const allocator = std.testing.allocator;

    const r1 = try CDO.roll1dn(allocator, 6);
    defer r1.deinit(allocator);

    inline for (1..4) |k| {
        inline for (1..3) |d| {
            const result = try CDO.rollKTimesDropLow(allocator, r1, @intCast(k), @intCast(d));
            defer result.deinit(allocator);

            const brute_result = try generate_distr_by_brute_force(allocator, struct {
                pub fn f(v: []const u64) u64 {
                    var sorted = allocator.alloc(u64, v.len) catch unreachable;
                    defer allocator.free(sorted);
                    @memcpy(sorted, v);
                    std.mem.sort(u64, sorted, {}, std.sort.asc(u64));

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

test "CountDiceOutcomes - roll k times drop lowest d - fuzz test deallocations under errors" {
    const CDO = CountDiceOutcomes(u32, usize, u8);

    const test_fn = struct {
        fn func(
            allocator: std.mem.Allocator,
            roll1: SequenceWithOffset(usize, u8),
            k: u32,
            d: u32,
        ) !void {
            const result = CDO.rollKTimesDropLow(allocator, roll1, k, d) catch |err| switch (err) {
                error.Overflow => return {},
                error.OutOfMemory => return error.OutOfMemory,
            };
            defer result.deinit(allocator);
        }
    }.func;

    const allocator = std.testing.allocator;
    var prng: std.Random.DefaultPrng = .init(std.testing.random_seed);
    for (0..100) |_| {
        const roll1: SequenceWithOffset(usize, u8) = try generateRngIntSeq(
            prng.random(),
            allocator,
            usize,
            u8,
            0,
            10,
            1,
            20,
            std.math.minInt(u8),
            std.math.maxInt(u8),
        );
        defer roll1.deinit(allocator);

        const k =
            // smith.valueRangeAtMost(u32, 1, 5);
            prng.random().intRangeAtMost(u32, 1, 5);
        const d =
            // smith.valueRangeAtMost(u32, 1, 5);
            prng.random().intRangeAtMost(u32, 1, 3);

        try std.testing.checkAllAllocationFailures(allocator, test_fn, .{ roll1, k, d });
    }
}

test "CountDiceOutcomes - roll4d6 drop lowest 1 deallocates on error" {
    const CDO = CountDiceOutcomes(u32, usize, u64);
    const allocator = std.testing.allocator;

    const test_fn = struct {
        pub fn func(alloc: std.mem.Allocator) !void {
            const roll1 = try CDO.roll1dn(alloc, 6);
            defer roll1.deinit(alloc);

            const result = try CDO.rollKTimesDropLow(alloc, roll1, 3, 1);
            defer result.deinit(alloc);
        }
    }.func;

    try std.testing.checkAllAllocationFailures(allocator, test_fn, .{});
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

test "CountDiceOutcomes - roll k times drop highest d - fuzz test deallocations under errors" {
    const CDO = CountDiceOutcomes(u32, usize, u8);

    const test_fn = struct {
        fn func(
            allocator: std.mem.Allocator,
            roll1: SequenceWithOffset(usize, u8),
            k: u32,
            d: u32,
        ) !void {
            const result = CDO.rollKTimesDropHigh(allocator, roll1, k, d) catch |err| switch (err) {
                error.Overflow => return {},
                error.OutOfMemory => return error.OutOfMemory,
            };
            defer result.deinit(allocator);
        }
    }.func;

    const allocator = std.testing.allocator;
    var prng: std.Random.DefaultPrng = .init(std.testing.random_seed);
    for (0..100) |_| {
        const roll1: SequenceWithOffset(usize, u8) = try generateRngIntSeq(
            prng.random(),
            allocator,
            usize,
            u8,
            0,
            10,
            1,
            20,
            std.math.minInt(u8),
            std.math.maxInt(u8),
        );
        defer roll1.deinit(allocator);

        const k =
            // smith.valueRangeAtMost(u32, 1, 5);
            prng.random().intRangeAtMost(u32, 1, 5);
        const d =
            // smith.valueRangeAtMost(u32, 1, 5);
            prng.random().intRangeAtMost(u32, 1, 3);
        try std.testing.checkAllAllocationFailures(allocator, test_fn, .{ roll1, k, d });
    }
}

test "CountDiceOutcomes - roll4d6 drop highest 1 deallocates on error" {
    const CDO = CountDiceOutcomes(u32, usize, u64);
    const allocator = std.testing.allocator;

    const test_fn = struct {
        pub fn func(alloc: std.mem.Allocator) !void {
            const roll1 = try CDO.roll1dn(alloc, 6);
            defer roll1.deinit(alloc);

            const result = try CDO.rollKTimesDropHigh(alloc, roll1, 3, 1);
            defer result.deinit(alloc);
        }
    }.func;

    try std.testing.checkAllAllocationFailures(allocator, test_fn, .{});
}

test "CountDiceOutcomes - roll kd6 = drop lowest 0 = drop highest 0" {
    const CDO = CountDiceOutcomes(u32, usize, u64);
    const allocator = std.testing.allocator;

    const r1 = try CDO.roll1dn(allocator, 6);
    defer r1.deinit(allocator);

    for (1..11) |k_usize| {
        const k: u32 = @intCast(k_usize);

        const result_flat = try CDO.rollKTimes(allocator, r1, k);
        defer result_flat.deinit(allocator);
        const result_drop_low = try CDO.rollKTimesDropLow(allocator, r1, k, 0);
        defer result_drop_low.deinit(allocator);
        const result_drop_high = try CDO.rollKTimesDropHigh(allocator, r1, k, 0);
        defer result_drop_high.deinit(allocator);

        try std.testing.expectEqual(result_flat.index_first, result_drop_low.index_first);
        try std.testing.expectEqualSlices(u64, result_flat.seq, result_drop_low.seq);

        try std.testing.expectEqual(result_flat.index_first, result_drop_high.index_first);
        try std.testing.expectEqualSlices(u64, result_flat.seq, result_drop_high.seq);
    }
}

fn generate_distr_by_brute_force(
    allocator: std.mem.Allocator,
    mapFn: fn (v: []const u64) u64,
    k: u32,
    n: u64,
) !SequenceWithOffset(usize, u64) {
    var values = std.ArrayList(u64).empty;

    const buffer = try allocator.alloc(u64, k);
    defer allocator.free(buffer);
    var iter = NestedRangeIterator.init(buffer, n - 1);
    while (true) {
        const value = mapFn(iter.get());
        const value_usize = @as(usize, @intCast(value));
        if (value_usize >= values.items.len) {
            try values.appendNTimes(allocator, 0, 1 + value_usize - values.items.len);
        }
        values.items[value_usize] += 1;

        if (!iter.increment()) break;
    }

    const i = for (0..values.items.len) |i| {
        if (values.items[i] != 0) break i;
    } else values.items.len;
    try values.replaceRange(allocator, 0, i, &[0]u64{});

    return .{ .index_first = i, .seq = try values.toOwnedSlice(allocator) };
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

    return struct {
        pub fn roll0(allocator: std.mem.Allocator) std.mem.Allocator.Error!SeqWOffset {
            return SeqWOffset.initSingle(allocator, 0, 1);
        }

        pub fn roll1dn(allocator: std.mem.Allocator, n: X) std.mem.Allocator.Error!SeqWOffset {
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
            errdefer result.deinit(allocator);
            var result_tmp: SeqWOffset = undefined;

            for (0..@intCast(k)) |_| {
                result_tmp = try result.addDistr(allocator, roll1);
                result.deinit(allocator);

                result = result_tmp;
                result_tmp = undefined;
            }

            return result;
        }

        pub fn rollKTimesDropLow(
            allocator: std.mem.Allocator,
            roll1: SeqWOffset,
            keep_count: D,
            drop_count: D,
        ) (std.mem.Allocator.Error || error{Overflow})!SeqWOffset {
            std.mem.reverse(Y, roll1.seq);
            defer std.mem.reverse(Y, roll1.seq);

            const result = try rollKTimesDropHigh(allocator, roll1, keep_count, drop_count);
            std.mem.reverse(Y, result.seq);
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
        pub fn mul2add3(_: void, x: i32) i64 {
            return @intCast(2 * x + 3);
        }
    }.mul2add3;

    const result = try seq.applyFnToValues(allocator, {}, mapFn);
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

test "SequenceWithOffset.toProb" {
    const SeqWOffset = SequenceWithOffset(usize, u32);
    const allocator = std.testing.allocator;

    var seq = SeqWOffset{
        .index_first = 3,
        .seq = @constCast(@as([]const u32, &.{ 2, 5, 4, 3, 2 })), // sum = 16 = 2^4
    };
    const result = try seq.toProb(f64, allocator);
    defer result.deinit(allocator);

    try std.testing.expectEqualSlices(
        f64,
        &.{ 0.125, 0.3125, 0.25, 0.1875, 0.125 },
        result.seq,
    );
}

test "SequenceWithOffset.toProb errors on sum larger than int max" {
    const SeqWOffset = SequenceWithOffset(usize, u8);
    const allocator = std.testing.allocator;

    var seq = SeqWOffset{
        .index_first = 3,
        .seq = @constCast(@as([]const u8, &.{ 255, 255 })),
    };
    try std.testing.expectEqual(error.Overflow, seq.toProb(f64, allocator));
}

test "SequenceWithOffset.toProb errors on sum larger than float resolution" {
    const SeqWOffset = SequenceWithOffset(usize, u64);
    const allocator = std.testing.allocator;

    var seq = SeqWOffset{
        .index_first = 3,
        .seq = @constCast(@as([]const u64, &.{std.math.maxInt(u64)})),
    };
    try std.testing.expectEqual(error.FloatOverflow, seq.toProb(f16, allocator));
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
            const buffer = try allocator.alloc(Y, std.math.sub(usize, self.seq.len + other.seq.len, 1) catch 0);
            errdefer allocator.free(buffer);

            return .{
                .index_first = self.index_first + other.index_first,
                .seq = try convolve1d(
                    Y,
                    buffer,
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
            context: anytype,
            mapFn: anytype,
        ) std.mem.Allocator.Error!SequenceWithOffset(X, @TypeOf(mapFn(context, self.seq[0]))) {
            // const YNew = comptime switch (@typeInfo(@TypeOf(mapFn))) {
            //     .Fn => |info| info.type orelse unreachable,
            //     else => unreachable,
            // };
            var buffer = try allocator.alloc(@TypeOf(mapFn(context, self.seq[0])), self.seq.len);
            for (0..buffer.len) |i| {
                buffer[i] = mapFn(context, self.seq[i]);
            }
            return .{ .index_first = self.index_first, .seq = buffer };
        }

        pub fn toProb(
            self: Self,
            comptime F: type,
            allocator: std.mem.Allocator,
        ) (std.mem.Allocator.Error || error{
            Overflow,
            FloatOverflow,
        })!SequenceWithOffset(X, F) {
            @setFloatMode(.optimized);

            var sum: Y = 0;
            for (self.seq) |x| {
                sum = try std.math.add(Y, sum, x);
            }
            const sum_float = @as(F, @floatFromInt(sum));
            if (!std.math.isFinite(sum_float)) {
                return error.FloatOverflow;
            }

            const convertValue = struct {
                fn func(factor: F, y: Y) F {
                    return @as(F, @floatFromInt(y)) * factor;
                }
            }.func;

            return self.applyFnToValues(allocator, 1.0 / sum_float, convertValue);
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
    std.debug.assert(result.len == std.math.sub(usize, a1.len + a2.len, 1) catch 0);

    for (result, 0..) |_, i| {
        const a1sub = a1[std.math.sub(usize, i + 1, a2.len) catch 0 .. @min(a1.len, i + 1)];
        const a2sub = a2[std.math.sub(usize, i + 1, a1.len) catch 0 .. @min(a2.len, i + 1)];
        std.debug.assert(a1sub.len == a2sub.len);

        result[i] = 0;
        for (0..a1sub.len) |j| {
            result[i] = try std.math.add(
                T,
                result[i],
                try std.math.mul(T, a1sub[j], a2sub[a1sub.len - j - 1]),
            );
        }
    }

    return result;
}

fn binomial(T: type, n: T, k: T) !T {
    if (k > n - k) {
        return binomial(T, n, n - k);
    }
    var result: T = 1;
    for (0..k) |_ki| {
        const ki: T = @intCast(_ki);
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
