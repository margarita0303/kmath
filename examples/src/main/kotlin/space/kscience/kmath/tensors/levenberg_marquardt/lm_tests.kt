/*
 * Copyright 2018-2021 KMath contributors.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the license/LICENSE.txt file.
 */

package space.kscience.kmath.tensors.levenberg_marquardt

import space.kscience.kmath.nd.as2D
import space.kscience.kmath.tensors.core.BroadcastDoubleTensorAlgebra
import space.kscience.kmath.tensors.core.BroadcastDoubleTensorAlgebra.eq
import space.kscience.kmath.tensors.core.BroadcastDoubleTensorAlgebra.fromArray
import space.kscience.kmath.tensors.core.BroadcastDoubleTensorAlgebra.ones
import kotlin.math.roundToInt

fun test_lm_func() {
    var t = ones(intArrayOf(100, 1)).as2D()
    for (i in 0 until 100) {
        t[i, 0] = t[i, 0] * (i + 1)
    }

    val p_true_example_number_1 = fromArray(intArrayOf(4, 1), doubleArrayOf(20.0, 10.0, 1.0, 50.0)).as2D()
    val p_true_example_number_2 = fromArray(intArrayOf(4, 1), doubleArrayOf(20.0, -24.0, 30.0, -40.0)).as2D()
    val p_true_example_number_3 = fromArray(intArrayOf(4, 1), doubleArrayOf(6.0, 20.0, 1.0, 5.0)).as2D()

    example_number = 1
    val y_dat_example_number_1 = lm_func(t, p_true_example_number_1)
    example_number = 2
    val y_dat_example_number_2 = lm_func(t, p_true_example_number_2)
    example_number = 3
    val y_dat_example_number_3 = lm_func(t, p_true_example_number_3)

    val buffer_y_expected_example_number_1 = lm_func_test1_buffer_y_expected
    val buffer_y_expected_example_number_2 = lm_func_test2_buffer_y_expected
    val buffer_y_expected_example_number_3 = lm_func_test3_buffer_y_expected

    val y_dat_expected_example_number_1 = BroadcastDoubleTensorAlgebra.fromArray(intArrayOf(100, 1), buffer_y_expected_example_number_1).as2D()
    val y_dat_expected_example_number_2 = BroadcastDoubleTensorAlgebra.fromArray(intArrayOf(100, 1), buffer_y_expected_example_number_2).as2D()
    val y_dat_expected_example_number_3 = BroadcastDoubleTensorAlgebra.fromArray(intArrayOf(100, 1), buffer_y_expected_example_number_3).as2D()
    for (i in 0 until 100) {
        y_dat_example_number_1[i, 0] = (y_dat_example_number_1[i, 0] * 10000.0).roundToInt() / 10000.0
        y_dat_example_number_2[i, 0] = (y_dat_example_number_2[i, 0] * 10000.0).roundToInt() / 10000.0
        y_dat_example_number_3[i, 0] = (y_dat_example_number_3[i, 0] * 10000.0).roundToInt() / 10000.0
    }
    if (y_dat_example_number_1.eq(y_dat_expected_example_number_1) &&
        y_dat_example_number_2.eq(y_dat_expected_example_number_2) &&
        y_dat_example_number_3.eq(y_dat_expected_example_number_3)){
        println("Ok")
    }
    else {
        println("Fail")
    }
}
fun test_lm_Broyden_J() {
    val p_old = BroadcastDoubleTensorAlgebra.fromArray(intArrayOf(4, 1), doubleArrayOf(6.2790, 19.2436, 1.0349, 5.0141))
    val buffer_y_old = lm_Broyden_J_buffer_y_old
    val y_old = BroadcastDoubleTensorAlgebra.fromArray(intArrayOf(100, 1), buffer_y_old)

    val buffer_J = lm_Broyden_J_buffer_J
    val J = BroadcastDoubleTensorAlgebra.fromArray(intArrayOf(100, 4), buffer_J)
    val p = BroadcastDoubleTensorAlgebra.fromArray(
        intArrayOf(4, 1), doubleArrayOf(
            6.2757,
            19.2627,
            1.0352,
            5.0141
        )
    )

    val buffer_y = lm_Broyden_J_buffer_y
    val y = BroadcastDoubleTensorAlgebra.fromArray(intArrayOf(100, 1), buffer_y)

    val buffer_expected = lm_Broyden_J_buffer_expected

    val ans = lm_Broyden_J(p_old.as2D(), y_old.as2D(), J.as2D(), p.as2D(), y.as2D())
    val expected = fromArray(intArrayOf(100, 4), buffer_expected).as2D()

    if (ans.eq(expected)) {
        println("Ok")
    }
    else {
        println("Fail")
    }
}

fun test_lm_FD_J() {
    example_number = 1
    var t = ones(intArrayOf(100, 1)).as2D()
    for (i in 0 until 100) {
        t[i, 0] = t[i, 0] * (i + 1)
    }

    val p = BroadcastDoubleTensorAlgebra.fromArray(
        intArrayOf(4, 1), doubleArrayOf(5.0, 2.0, 0.2, 10.0)
    ).as2D()

    val y_hat = feval(::lm_func, t, p)
    func_calls += 1

    val dp = BroadcastDoubleTensorAlgebra.fromArray(
        intArrayOf(4, 1), DoubleArray(4) { -0.01 }
    ).as2D()

    val ans = lm_FD_J(::lm_func, t, p, y_hat, dp)
    val expected = BroadcastDoubleTensorAlgebra.fromArray(
        intArrayOf(100, 4), lm_FD_J_J
    ).as2D()

    for (i in 0 until 100) {
        for (j in 0 until 4) {
            ans[i, j] = (ans[i, j] * 10000.0).roundToInt() / 10000.0
        }
    }

    if (ans.eq(expected)) {
        println("Ok")
    }
    else {
        println("Fail")
    }
}

fun test_lm_matx() {
    var t = ones(intArrayOf(100, 1)).as2D()
    for (i in 0 until 100) {
        t[i, 0] = t[i, 0] * (i + 1)
    }
    val p_old = BroadcastDoubleTensorAlgebra.fromArray(
        intArrayOf(4, 1), DoubleArray(4) { 0.0 }
    ).as2D()

    val y_old = BroadcastDoubleTensorAlgebra.fromArray(
        intArrayOf(100, 1), DoubleArray(100) { 0.0 }
    ).as2D()

    val J = BroadcastDoubleTensorAlgebra.fromArray(
        intArrayOf(100, 4), DoubleArray(400) { 0.0 }
    ).as2D()

    val p = BroadcastDoubleTensorAlgebra.fromArray(
        intArrayOf(4, 1), doubleArrayOf(5.0, 2.0, 0.2, 10.0)
    ).as2D()

    val y_dat = BroadcastDoubleTensorAlgebra.fromArray(
        intArrayOf(100, 1), lm_matx_y_dat
    ).as2D()

    val weight = BroadcastDoubleTensorAlgebra.fromArray(
        intArrayOf(100, 1), DoubleArray(100) { 4.0 }
    ).as2D()

    val dp = BroadcastDoubleTensorAlgebra.fromArray(
        intArrayOf(4, 1), DoubleArray(4) { -0.01 }
    ).as2D()

    example_number = 1
    lm_matx(::lm_func, t, p_old, y_old, 1, J, p, y_dat, weight, dp)
}

fun test_lm() {
    example_number = 1
    val p_init = BroadcastDoubleTensorAlgebra.fromArray(
        intArrayOf(4, 1), doubleArrayOf(5.0, 2.0, 0.2, 10.0)
    ).as2D()

    var t = ones(intArrayOf(100, 1)).as2D()
    for (i in 0 until 100) {
        t[i, 0] = t[i, 0] * (i + 1)
    }

    val y_dat = BroadcastDoubleTensorAlgebra.fromArray(
        intArrayOf(100, 1), lm_matx_y_dat
    ).as2D()

    val weight = BroadcastDoubleTensorAlgebra.fromArray(
        intArrayOf(1, 1), DoubleArray(1) { 4.0 }
    ).as2D()

    val dp = BroadcastDoubleTensorAlgebra.fromArray(
        intArrayOf(1, 1), DoubleArray(1) { -0.01 }
    ).as2D()

    val p_min = BroadcastDoubleTensorAlgebra.fromArray(
        intArrayOf(4, 1), doubleArrayOf(-50.0, -20.0, -2.0, -100.0)
    ).as2D()

    val p_max = BroadcastDoubleTensorAlgebra.fromArray(
        intArrayOf(4, 1), doubleArrayOf(50.0, 20.0, 2.0, 100.0)
    ).as2D()

    val consts = BroadcastDoubleTensorAlgebra.fromArray(
        intArrayOf(1, 1), doubleArrayOf(0.0)
    ).as2D()

    val opts = doubleArrayOf(3.0, 100.0, 1e-3, 1e-3, 1e-1, 1e-1, 1e-2, 11.0, 9.0, 1.0)

    lm(::lm_func, p_init, t, y_dat, weight, dp, p_min, p_max, consts, opts, 10)
}