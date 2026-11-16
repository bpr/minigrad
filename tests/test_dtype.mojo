from std.testing import TestSuite, assert_equal, assert_true, assert_false

from minigrad.dtype import DT, parse_dtype, promote, sum_acc_dtype


def test_itemsize() raises:
    assert_equal(DT.bool.itemsize(), 1)
    assert_equal(DT.int16.itemsize(), 2)
    assert_equal(DT.uint32.itemsize(), 4)
    assert_equal(DT.float32.itemsize(), 4)
    assert_equal(DT.int64.itemsize(), 8)
    assert_equal(DT.float64.itemsize(), 8)


def test_promote() raises:
    assert_equal(promote(DT.bool, DT.int8), DT.int8)
    assert_equal(promote(DT.int8, DT.uint8), DT.int16)
    assert_equal(promote(DT.uint8, DT.int32), DT.int32)
    assert_equal(promote(DT.uint32, DT.int32), DT.int64)
    assert_equal(promote(DT.uint64, DT.int64), DT.float32)
    assert_equal(promote(DT.int64, DT.float32), DT.float32)
    assert_equal(promote(DT.float32, DT.float64), DT.float64)
    assert_equal(promote(DT.uint16, DT.uint64), DT.uint64)
    for a in range(DT.COUNT):
        for b in range(DT.COUNT):
            assert_equal(promote(DT(a), DT(b)), promote(DT(b), DT(a)))
            assert_equal(promote(DT(a), DT(a)), DT(a))


def test_names() raises:
    assert_equal(String(DT.float32), "float32")
    assert_true(DT.uint8.is_unsigned())
    assert_false(DT.int8.is_unsigned())
    assert_equal(sum_acc_dtype(DT.bool), DT.int32)
    assert_equal(sum_acc_dtype(DT.uint8), DT.uint32)
    assert_equal(sum_acc_dtype(DT.float32), DT.float32)


def test_half() raises:
    assert_equal(DT.float16.itemsize(), 2)
    assert_equal(DT.bfloat16.itemsize(), 2)
    assert_true(DT.bfloat16.is_float() and DT.bfloat16.is_half())
    assert_false(DT.float32.is_half())
    assert_equal(promote(DT.float16, DT.bfloat16), DT.float32)
    assert_equal(promote(DT.float16, DT.float32), DT.float32)
    assert_equal(promote(DT.bfloat16, DT.int32), DT.bfloat16)
    assert_equal(promote(DT.float16, DT.float64), DT.float64)
    assert_equal(sum_acc_dtype(DT.float16), DT.float32)
    assert_equal(parse_dtype("bfloat16"), DT.bfloat16)
    for c in range(DT.COUNT):
        assert_equal(parse_dtype(DT(c).name()), DT(c))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
