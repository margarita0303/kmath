/*
 * Copyright 2018-2021 KMath contributors.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the license/LICENSE.txt file.
 */

package space.kscience.kmath.tensors.levenberg_marquardt

import space.kscience.kmath.misc.PerformancePitfall
import space.kscience.kmath.nd.MutableStructure2D
import space.kscience.kmath.nd.Structure2D
import space.kscience.kmath.tensors.core.tensorAlgebra
import space.kscience.kmath.tensors.core.withBroadcast
import kotlin.math.roundToInt

fun MutableStructure2D<Double>.print() {
    val n = this.shape.component1()
    val m = this.shape.component2()
    for (i in 0 until n) {
        for (j in 0 until m) {
            val x = (this[i, j] * 10000).roundToInt() / 10000.0
            print("$x ")
        }
        println()
    }
    println("______________")
}

fun Structure2D<Double>.print() {
    val n = this.shape.component1()
    val m = this.shape.component2()
    for (i in 0 until n) {
        for (j in 0 until m) {
            val x = (this[i, j] * 10000).roundToInt() / 10000.0
            print("$x ")
        }
        println()
    }
    println("______________")
}

@OptIn(PerformancePitfall::class)
fun main(): Unit = Double.tensorAlgebra.withBroadcast {
    test_lm()

//    Expected (matlab output):
//    >  1:  6 | chi_sq= 1.143e+03 | lambda= 1.1e-01
//    param:    5.000e+00  2.000e+00  2.000e-01  1.000e+01
//    dp/p :   -4.196e+00  1.926e+01 -2.488e+01  3.286e+01
//    >  2:  7 | chi_sq= 1.143e+03 | lambda= 1.2e+00
//    param:    5.000e+00  2.000e+00  2.000e-01  1.000e+01
//    dp/p :    1.947e+00  6.879e+00 -1.786e+00  2.013e+01
//    >  3:  8 | chi_sq= 1.143e+03 | lambda= 1.3e+01
//    param:    5.000e+00  2.000e+00  2.000e-01  1.000e+01
//    dp/p :    1.964e+00  2.800e+00  8.197e+00  7.982e+00
//    >  4: 10 | chi_sq= 7.334e+02 | lambda= 1.5e+00
//    param:    7.627e+00  3.534e+00  6.084e-01  2.463e+01
//    dp/p :    3.444e-01  4.341e-01  6.712e-01  5.940e-01
//    >  5: 12 | chi_sq= 1.754e+02 | lambda= 1.6e-01
//    param:    2.721e+01  7.796e+00  1.593e+00  4.528e+01
//    dp/p :    7.197e-01  5.466e-01  6.181e-01  4.560e-01
//    >  6: 14 | chi_sq= 7.760e+01 | lambda= 1.8e-02
//    param:    2.106e+01  6.872e+00  1.011e+00  3.954e+01
//    dp/p :   -2.918e-01 -1.344e-01 -5.762e-01 -1.452e-01
//    >  7: 15 | chi_sq= 1.754e+02 | lambda= 2.0e-01
//    param:    2.106e+01  6.872e+00  1.011e+00  3.954e+01
//    dp/p :    1.840e-01 -8.024e-02 -5.539e-02  3.720e-02
//    >  8: 21 | chi_sq= 5.361e+01 | lambda= 2.2e-02
//    param:    2.351e+01  8.055e+00  9.908e-01  4.149e+01
//    dp/p :    1.041e-01  1.469e-01 -1.992e-02  4.709e-02
//    >  9: 23 | chi_sq= 1.004e+00 | lambda= 2.5e-03
//    param:    2.099e+01  8.966e+00  1.028e+00  4.899e+01
//    dp/p :   -1.201e-01  1.016e-01  3.620e-02  1.530e-01
//    > 10: 25 | chi_sq= 9.602e-01 | lambda= 2.8e-04
//    param:    2.043e+01  9.633e+00  1.002e+00  4.979e+01
//    dp/p :   -2.718e-02  6.919e-02 -2.608e-02  1.609e-02
//    > 11: 27 | chi_sq= 9.038e-01 | lambda= 3.1e-05
//    param:    2.052e+01  9.849e+00  9.970e-01  5.020e+01
//    dp/p :    4.143e-03  2.199e-02 -4.859e-03  8.294e-03
//    > 12: 29 | chi_sq= 9.037e-01 | lambda= 3.4e-06
//    param:    2.053e+01  9.833e+00  9.976e-01  5.017e+01
//    dp/p :    4.917e-04 -1.628e-03  5.679e-04 -5.725e-04
//    > 13: 31 | chi_sq= 9.037e-01 | lambda= 3.8e-07
//    param:    2.053e+01  9.834e+00  9.976e-01  5.017e+01
//    dp/p :   -1.850e-05  2.924e-05 -3.032e-05  8.440e-06
}