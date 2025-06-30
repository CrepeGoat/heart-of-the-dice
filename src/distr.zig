const std = @import("std");

pub fn Data(X: type, Y: type) type {
    return struct {
        x: X,
        y: Y,
    };
}

test "convolve two arrays" {
    const a1 = [_]i32{ 1, 2, 3 };
    const a2 = [_]i32{ 5, 7 };

    var result = [1]i32{999} ** 5;
    const calc_result = convolve1d(i32, &a1, &a2, &result);

    try std.testing.expectEqual(&result[0], &calc_result[0]);
    try std.testing.expectEqualSlices(i32, &[_]i32{ 5, 17, 29, 21 }, calc_result);
    try std.testing.expectEqualSlices(i32, &[1]i32{999}, result[4..]);
}

fn convolve1d(comptime T: type, a1: []const T, a2: []const T, result: []T) []T {
    const convolve_len = a1.len + a2.len - 1;
    if (result.len < convolve_len) {
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
