/*
 * Copyright 2018-2021 KMath contributors.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the license/LICENSE.txt file.
 */

package space.kscience.kmath.tensors.levenberg_marquardt

import space.kscience.kmath.nd.MutableStructure2D
import space.kscience.kmath.nd.as2D
import space.kscience.kmath.nd4j.DoubleNd4jTensorAlgebra.max
import space.kscience.kmath.real.pow
import space.kscience.kmath.real.times
import space.kscience.kmath.tensors.core.BroadcastDoubleTensorAlgebra
import space.kscience.kmath.tensors.core.BroadcastDoubleTensorAlgebra.div
import space.kscience.kmath.tensors.core.BroadcastDoubleTensorAlgebra.dot
import space.kscience.kmath.tensors.core.BroadcastDoubleTensorAlgebra.exp
import space.kscience.kmath.tensors.core.BroadcastDoubleTensorAlgebra.minus
import space.kscience.kmath.tensors.core.BroadcastDoubleTensorAlgebra.ones
import space.kscience.kmath.tensors.core.BroadcastDoubleTensorAlgebra.sin
import space.kscience.kmath.tensors.core.BroadcastDoubleTensorAlgebra.times
import space.kscience.kmath.tensors.core.BroadcastDoubleTensorAlgebra.transpose
import space.kscience.kmath.tensors.core.BroadcastDoubleTensorAlgebra.zeros
import space.kscience.kmath.tensors.core.DoubleTensorAlgebra.Companion.copy
import space.kscience.kmath.tensors.core.DoubleTensorAlgebra.Companion.plus

/* matrix -> column of all elemnets */
fun make_column(tensor: MutableStructure2D<Double>) : MutableStructure2D<Double> {
    val shape = intArrayOf(tensor.shape.component1() * tensor.shape.component2(), 1)
    var buffer = DoubleArray(tensor.shape.component1() * tensor.shape.component2())
    for (i in 0 until tensor.shape.component1()) {
        for (j in 0 until tensor.shape.component2()) {
            buffer[i * tensor.shape.component2() + j] = tensor[i, j]
        }
    }
    var column = BroadcastDoubleTensorAlgebra.fromArray(shape, buffer).as2D()
    return column
}

/* column length */
fun length(column: MutableStructure2D<Double>) : Int {
    return column.shape.component1()
}

fun MutableStructure2D<Double>.abs() {
    for (i in 0 until this.shape.component1()) {
        for (j in 0 until this.shape.component2()) {
            this[i, j] = kotlin.math.abs(this[i, j])
        }
    }
}

fun abs(input: MutableStructure2D<Double>): MutableStructure2D<Double> {
    val tensor = ones(intArrayOf(input.shape.component1(), input.shape.component2())).as2D()
    for (i in 0 until tensor.shape.component1()) {
        for (j in 0 until tensor.shape.component2()) {
            tensor[i, j] = kotlin.math.abs(input[i, j])
        }
    }
    return tensor
}

fun diag(input: MutableStructure2D<Double>): MutableStructure2D<Double> {
    val tensor = ones(intArrayOf(input.shape.component1(), 1)).as2D()
    for (i in 0 until tensor.shape.component1()) {
        tensor[i, 0] = input[i, i]
    }
    return tensor
}

fun make_matrx_with_diagonal(column: MutableStructure2D<Double>): MutableStructure2D<Double> {
    val size = column.shape.component1()
    val tensor = zeros(intArrayOf(size, size)).as2D()
    for (i in 0 until size) {
        tensor[i, i] = column[i, 0]
    }
    return tensor
}

fun lm_eye(size: Int): MutableStructure2D<Double> {
    val column = ones(intArrayOf(size, 1)).as2D()
    return make_matrx_with_diagonal(column)
}

fun largest_element_comparison(a: MutableStructure2D<Double>, b: MutableStructure2D<Double>): MutableStructure2D<Double> {
    val a_sizeX = a.shape.component1()
    val a_sizeY = a.shape.component2()
    val b_sizeX = b.shape.component1()
    val b_sizeY = b.shape.component2()
    val tensor = zeros(intArrayOf(kotlin.math.max(a_sizeX, b_sizeX), kotlin.math.max(a_sizeY, b_sizeY))).as2D()
    for (i in 0 until tensor.shape.component1()) {
        for (j in 0 until tensor.shape.component2()) {
            if (i < a_sizeX && i < b_sizeX && j < a_sizeY && j < b_sizeY) {
                tensor[i, j] = kotlin.math.max(a[i, j], b[i, j])
            }
            else if (i < a_sizeX && j < a_sizeY) {
                tensor[i, j] = a[i, j]
            }
            else {
                tensor[i, j] = b[i, j]
            }
        }
    }
    return tensor
}

fun smallest_element_comparison(a: MutableStructure2D<Double>, b: MutableStructure2D<Double>): MutableStructure2D<Double> {
    val a_sizeX = a.shape.component1()
    val a_sizeY = a.shape.component2()
    val b_sizeX = b.shape.component1()
    val b_sizeY = b.shape.component2()
    val tensor = zeros(intArrayOf(kotlin.math.max(a_sizeX, b_sizeX), kotlin.math.max(a_sizeY, b_sizeY))).as2D()
    for (i in 0 until tensor.shape.component1()) {
        for (j in 0 until tensor.shape.component2()) {
            if (i < a_sizeX && i < b_sizeX && j < a_sizeY && j < b_sizeY) {
                tensor[i, j] = kotlin.math.min(a[i, j], b[i, j])
            }
            else if (i < a_sizeX && j < a_sizeY) {
                tensor[i, j] = a[i, j]
            }
            else {
                tensor[i, j] = b[i, j]
            }
        }
    }
    return tensor
}

fun get_zero_indices(column: MutableStructure2D<Double>, epsilon: Double = 0.000001): MutableStructure2D<Double>? {
    var idx = emptyArray<Double>()
    for (i in 0 until column.shape.component1()) {
        if (kotlin.math.abs(column[i, 0]) > epsilon) {
            idx += (i + 1.0)
        }
    }
    if (idx.size > 0) {
        return BroadcastDoubleTensorAlgebra.fromArray(intArrayOf(idx.size, 1), idx.toDoubleArray()).as2D()
    }
    return null
}

fun feval(func: (MutableStructure2D<Double>,  MutableStructure2D<Double>) ->  MutableStructure2D<Double>,
          t: MutableStructure2D<Double>, p: MutableStructure2D<Double>)
        : MutableStructure2D<Double>
{
    return func(t, p)
}

fun lm_matx(func: (MutableStructure2D<Double>, MutableStructure2D<Double>) -> MutableStructure2D<Double>,
            t: MutableStructure2D<Double>, p_old: MutableStructure2D<Double>, y_old: MutableStructure2D<Double>,
            dX2: Int, J_input: MutableStructure2D<Double>, p: MutableStructure2D<Double>,
            y_dat: MutableStructure2D<Double>, weight: MutableStructure2D<Double>, dp:MutableStructure2D<Double>) : Array<MutableStructure2D<Double>>
{
    // default: dp = 0.001

    val Npnt = length(y_dat)               // number of data points
    val Npar = length(p)                   // number of parameters

    val y_hat = feval(func, t, p)          // evaluate model using parameters 'p'
    func_calls += 1

    var J = J_input

    if (iteration % (2 * Npar) == 0 || dX2 > 0) {
        J = lm_FD_J(func, t, p, y_hat, dp).as2D() // finite difference
    }
    else {
        J = lm_Broyden_J(p_old, y_old, J, p, y_hat).as2D() // rank-1 update
    }

    val delta_y = y_dat.minus(y_hat)

    val Chi_sq = delta_y.transpose().dot( delta_y.times(weight) ).as2D()
    val JtWJ = J.transpose().dot ( J.times( weight.dot( ones(intArrayOf(1, Npar)) ) ) ).as2D()
    val JtWdy = J.transpose().dot( weight.times(delta_y) ).as2D()

    return arrayOf(JtWJ,JtWdy,Chi_sq,y_hat,J)
}

fun lm_Broyden_J(p_old: MutableStructure2D<Double>, y_old: MutableStructure2D<Double>, J_input: MutableStructure2D<Double>,
                 p: MutableStructure2D<Double>, y: MutableStructure2D<Double>): MutableStructure2D<Double> {

    var J = J_input.copy()

    val h = p.minus(p_old)
    val increase = y.minus(y_old).minus( J.dot(h) ).dot(h.transpose()).div( (h.transpose().dot(h)).as2D()[0, 0] )
    J = J.plus(increase)

    return J.as2D()
}

fun lm_FD_J(func: (MutableStructure2D<Double>, MutableStructure2D<Double>) -> MutableStructure2D<Double>,
            t: MutableStructure2D<Double>, p: MutableStructure2D<Double>, y: MutableStructure2D<Double>,
            dp: MutableStructure2D<Double>): MutableStructure2D<Double> {

    // default: dp = 0.001 * ones(1,n)

    val m = length(y)              // number of data points
    val n = length(p)              // number of parameters

    val ps = p.copy().as2D()
    val J = zeros(intArrayOf(m, n)).as2D()  // initialize Jacobian to Zero
    val del = zeros(intArrayOf(n, 1)).as2D()

    for (j in 0 until n) {

        del[j, 0] = dp[j, 0] * (1 + kotlin.math.abs(p[j, 0])) // parameter perturbation
        p[j, 0] = ps[j, 0] + del[j, 0]                        // perturb parameter p(j)

        val epsilon = 0.0000001
        if (kotlin.math.abs(del[j, 0]) > epsilon) {
            val y1 = feval(func, t, p)
            func_calls += 1

            if (dp[j, 0] < 0) { // backwards difference
                for (i in 0 until J.shape.component1()) {
                    J[i, j] = (y1.as2D().minus(y).as2D())[i, 0] / del[j, 0]
                }
            }
            else {
                // Do tests for it
                println("Potential mistake")
                p[j, 0] = ps[j, 0] - del[j, 0] // central difference, additional func call
                for (i in 0 until J.shape.component1()) {
                    J[i, j] = (y1.as2D().minus(feval(func, t, p)).as2D())[i, 0] / (2 * del[j, 0])
                }
                func_calls += 1
            }
        }

        p[j, 0] = ps[j, 0] // restore p(j)
    }

    return J.as2D()
}

fun lm_func(t: MutableStructure2D<Double>, p: MutableStructure2D<Double>): MutableStructure2D<Double> {
    val m = t.shape.component1()
    var y_hat = zeros(intArrayOf(m , 1))

    if (example_number == 1) {
        y_hat = (t.times(-1.0 / p[1, 0])).exp().times(p[0, 0]) + t.times(p[2, 0]).times( (t.times(-1.0 / p[3, 0])).exp() )
    }
    else if (example_number == 2) {
        val mt = t.max()
        y_hat = (t.times(1.0 / mt)).times(p[0, 0]) +
                (t.times(1.0 / mt)).pow(2).times(p[1, 0]) +
                (t.times(1.0 / mt)).pow(3).times(p[2, 0]) +
                (t.times(1.0 / mt)).pow(4).times(p[3, 0])
    }
    else if (example_number == 3) {
        y_hat = (t.times(-1.0 / p[1, 0])).exp().times(p[0, 0]) + (t.times(1.0 / p[3, 0])).sin().times(p[2, 0])
    }

    return y_hat.as2D()
}