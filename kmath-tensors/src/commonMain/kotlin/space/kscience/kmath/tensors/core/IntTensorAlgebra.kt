/*
 * Copyright 2018-2022 KMath contributors.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the license/LICENSE.txt file.
 */


@file:OptIn(PerformancePitfall::class)

package space.kscience.kmath.tensors.core

import space.kscience.kmath.PerformancePitfall
import space.kscience.kmath.nd.*
import space.kscience.kmath.operations.IntRing
import space.kscience.kmath.structures.*
import space.kscience.kmath.tensors.api.*
import space.kscience.kmath.tensors.core.internal.*
import kotlin.math.*

/**
 * Implementation of basic operations over double tensors and basic algebra operations on them.
 */
public open class IntTensorAlgebra : TensorAlgebra<Int, IntRing> {

    public companion object : IntTensorAlgebra()


    override val elementAlgebra: IntRing get() = IntRing


    /**
     * Applies the [transform] function to each element of the tensor and returns the resulting modified tensor.
     *
     * @param transform the function to be applied to each element of the tensor.
     * @return the resulting tensor after applying the function.
     */
    @Suppress("OVERRIDE_BY_INLINE")
    final override inline fun StructureND<Int>.map(transform: IntRing.(Int) -> Int): IntTensor {
        val tensor = this.asIntTensor()
        //TODO remove additional copy
        val array = IntBuffer(tensor.source.size) { IntRing.transform(tensor.source[it]) }
        return IntTensor(
            tensor.shape,
            array,
        )
    }

    public inline fun Tensor<Int>.mapInPlace(operation: (Int) -> Int) {
        if (this is IntTensor) {
            source.mapInPlace(operation)
        } else {
            indices.forEach { set(it, operation(get(it))) }
        }
    }

    public inline fun Tensor<Int>.mapIndexedInPlace(operation: (IntArray, Int) -> Int) {
        indices.forEach { set(it, operation(it, get(it))) }
    }

    @Suppress("OVERRIDE_BY_INLINE")
    final override inline fun StructureND<Int>.mapIndexed(transform: IntRing.(index: IntArray, Int) -> Int): IntTensor {
        val tensor = this.asIntTensor()
        //TODO remove additional copy
        val buffer = IntBuffer(tensor.source.size) {
            IntRing.transform(tensor.indices.index(it), tensor.source[it])
        }
        return IntTensor(tensor.shape, buffer)
    }

    @Suppress("OVERRIDE_BY_INLINE")
    final override inline fun zip(
        left: StructureND<Int>,
        right: StructureND<Int>,
        transform: IntRing.(Int, Int) -> Int,
    ): IntTensor {
        checkShapesCompatible(left, right)

        val leftTensor = left.asIntTensor()
        val rightTensor = right.asIntTensor()
        val buffer = IntBuffer(leftTensor.source.size) {
            IntRing.transform(leftTensor.source[it], rightTensor.source[it])
        }
        return IntTensor(leftTensor.shape, buffer)
    }


    public inline fun StructureND<Int>.reduceElements(transform: (IntBuffer) -> Int): Int =
        transform(asIntTensor().source.copy())
    //TODO do we need protective copy?

    override fun StructureND<Int>.valueOrNull(): Int? {
        val dt = asIntTensor()
        return if (dt.shape contentEquals ShapeND(1)) dt.source[0] else null
    }

    override fun StructureND<Int>.value(): Int = valueOrNull()
        ?: throw IllegalArgumentException("The tensor shape is $shape, but value method is allowed only for shape [1]")

    /**
     * Constructs a tensor with the specified shape and data.
     *
     * @param shape the desired shape for the tensor.
     * @param array one-dimensional data array.
     * @return tensor with the [shape] shape and [array] data.
     */
    public fun fromArray(shape: ShapeND, array: IntArray): IntTensor {
        checkNotEmptyShape(shape)
        check(array.isNotEmpty()) { "Illegal empty buffer provided" }
        check(array.size == shape.linearSize) {
            "Inconsistent shape ${shape} for buffer of size ${array.size} provided"
        }
        return IntTensor(shape, array.asBuffer())
    }

    /**
     * Constructs a tensor with the specified shape and initializer.
     *
     * @param shape the desired shape for the tensor.
     * @param initializer mapping tensor indices to values.
     * @return tensor with the [shape] shape and data generated by the [initializer].
     */
    override fun structureND(shape: ShapeND, initializer: IntRing.(IntArray) -> Int): IntTensor = fromArray(
        shape,
        RowStrides(shape).asSequence().map { IntRing.initializer(it) }.toMutableList().toIntArray()
    )

    override fun Tensor<Int>.getTensor(i: Int): IntTensor {
        val dt = asIntTensor()
        val lastShape = shape.last(shape.size - 1)
        val newShape = if (lastShape.isNotEmpty()) lastShape else ShapeND(1)
        return IntTensor(newShape, dt.source.view(newShape.linearSize * i))
    }

    /**
     * Creates a tensor of a given shape and fills all elements with a given value.
     *
     * @param value the value to fill the output tensor with.
     * @param shape array of integers defining the shape of the output tensor.
     * @return tensor with the [shape] shape and filled with [value].
     */
    public fun full(value: Int, shape: ShapeND): IntTensor {
        checkNotEmptyShape(shape)
        val buffer = IntBuffer(shape.linearSize) { value }
        return IntTensor(shape, buffer)
    }

    /**
     * Returns a tensor with the same shape as `input` filled with [value].
     *
     * @param value the value to fill the output tensor with.
     * @return tensor with the `input` tensor shape and filled with [value].
     */
    public fun fullLike(structureND: StructureND<*>, value: Int): IntTensor {
        val shape = structureND.shape
        val buffer = IntBuffer(structureND.indices.linearSize) { value }
        return IntTensor(shape, buffer)
    }

    /**
     * Returns a tensor filled with the scalar value `0`, with the shape defined by the variable argument [shape].
     *
     * @param shape array of integers defining the shape of the output tensor.
     * @return tensor filled with the scalar value `0`, with the [shape] shape.
     */
    public fun zeros(shape: ShapeND): IntTensor = full(0, shape)

    /**
     * Returns a tensor filled with the scalar value `0`, with the same shape as a given array.
     *
     * @return tensor filled with the scalar value `0`, with the same shape as `input` tensor.
     */
    public fun zeroesLike(structureND: StructureND<Int>): IntTensor = fullLike(structureND.asIntTensor(), 0)

    /**
     * Returns a tensor filled with the scalar value `1`, with the shape defined by the variable argument [shape].
     *
     * @param shape array of integers defining the shape of the output tensor.
     * @return tensor filled with the scalar value `1`, with the [shape] shape.
     */
    public fun ones(shape: ShapeND): IntTensor = full(1, shape)

    /**
     * Returns a tensor filled with the scalar value `1`, with the same shape as a given array.
     *
     * @return tensor filled with the scalar value `1`, with the same shape as `input` tensor.
     */
    public fun onesLike(structureND: Tensor<*>): IntTensor = fullLike(structureND, 1)

    /**
     * Returns a 2D tensor with shape ([n], [n]), with ones on the diagonal and zeros elsewhere.
     *
     * @param n the number of rows and columns
     * @return a 2-D tensor with ones on the diagonal and zeros elsewhere.
     */
    public fun eye(n: Int): IntTensor {
        val shape = ShapeND(n, n)
        val buffer = IntBuffer(n * n) { 0 }
        val res = IntTensor(shape, buffer)
        for (i in 0 until n) {
            res[intArrayOf(i, i)] = 1
        }
        return res
    }

    override fun Int.plus(arg: StructureND<Int>): IntTensor = arg.map { this@plus + it }

    override fun StructureND<Int>.plus(arg: Int): IntTensor = map { it + arg }

    override fun StructureND<Int>.plus(arg: StructureND<Int>): IntTensor = zip(this, arg) { l, r -> l + r }

    override fun Tensor<Int>.plusAssign(value: Int) {
        mapInPlace { it + value }
    }

    override fun Tensor<Int>.plusAssign(arg: StructureND<Int>) {
        checkShapesCompatible(asIntTensor(), arg.asIntTensor())
        mapIndexedInPlace { index, value ->
            value + arg[index]
        }
    }

    override fun Int.minus(arg: StructureND<Int>): IntTensor = arg.map { this@minus - it }

    override fun StructureND<Int>.minus(arg: Int): IntTensor = map { it - arg }

    override fun StructureND<Int>.minus(arg: StructureND<Int>): IntTensor = zip(this, arg) { l, r -> l - r }

    override fun Tensor<Int>.minusAssign(value: Int) {
        mapInPlace { it - value }
    }

    override fun Tensor<Int>.minusAssign(arg: StructureND<Int>) {
        checkShapesCompatible(this, arg)
        mapIndexedInPlace { index, value -> value - arg[index] }
    }

    override fun Int.times(arg: StructureND<Int>): IntTensor = arg.map { this@times * it }

    override fun StructureND<Int>.times(arg: Int): IntTensor = arg * asIntTensor()

    override fun StructureND<Int>.times(arg: StructureND<Int>): IntTensor = zip(this, arg) { l, r -> l * r }

    override fun Tensor<Int>.timesAssign(value: Int) {
        mapInPlace { it * value }
    }

    override fun Tensor<Int>.timesAssign(arg: StructureND<Int>) {
        checkShapesCompatible(this, arg)
        mapIndexedInPlace { index, value -> value * arg[index] }
    }

    override fun StructureND<Int>.unaryMinus(): IntTensor = map { -it }

    override fun StructureND<Int>.transposed(i: Int, j: Int): Tensor<Int> {
        val actualI = if (i >= 0) i else shape.size + i
        val actualJ = if(j>=0) j else shape.size + j
        return asIntTensor().permute(
            shape.transposed(actualI, actualJ)
        ) { originIndex ->
            originIndex.copyOf().apply {
                val ith = get(actualI)
                val jth = get(actualJ)
                set(actualI, jth)
                set(actualJ, ith)
            }
        }
//        // TODO change strides instead of changing content
//        val dt = asIntTensor()
//        val ii = dt.minusIndex(i)
//        val jj = dt.minusIndex(j)
//        checkTranspose(dt.dimension, ii, jj)
//        val n = dt.linearSize
//        val resBuffer = IntArray(n)
//
//        val resShape = dt.shape.toArray()
//        resShape[ii] = resShape[jj].also { resShape[jj] = resShape[ii] }
//
//        val resTensor = IntTensor(Shape(resShape), resBuffer.asBuffer())
//
//        for (offset in 0 until n) {
//            val oldMultiIndex = dt.indices.index(offset)
//            val newMultiIndex = oldMultiIndex.copyOf()
//            newMultiIndex[ii] = newMultiIndex[jj].also { newMultiIndex[jj] = newMultiIndex[ii] }
//
//            val linearIndex = resTensor.indices.offset(newMultiIndex)
//            resTensor.source[linearIndex] = dt.source[offset]
//        }
//        return resTensor
    }

    override fun Tensor<Int>.view(shape: ShapeND): IntTensor {
        checkView(asIntTensor(), shape)
        return IntTensor(shape, asIntTensor().source)
    }

    override fun Tensor<Int>.viewAs(other: StructureND<Int>): IntTensor =
        view(other.shape)

    override fun StructureND<Int>.dot(other: StructureND<Int>): IntTensor {
        TODO("not implemented for integers")
    }

    override fun diagonalEmbedding(
        diagonalEntries: StructureND<Int>,
        offset: Int,
        dim1: Int,
        dim2: Int,
    ): IntTensor {
        val n = diagonalEntries.shape.size
        val d1 = minusIndexFrom(n + 1, dim1)
        val d2 = minusIndexFrom(n + 1, dim2)

        check(d1 != d2) {
            "Diagonal dimensions cannot be identical $d1, $d2"
        }
        check(d1 <= n && d2 <= n) {
            "Dimension out of range"
        }

        var lessDim = d1
        var greaterDim = d2
        var realOffset = offset
        if (lessDim > greaterDim) {
            realOffset *= -1
            lessDim = greaterDim.also { greaterDim = lessDim }
        }

        val resShape = diagonalEntries.shape.slice(0 until lessDim) +
                intArrayOf(diagonalEntries.shape[n - 1] + abs(realOffset)) +
                diagonalEntries.shape.slice(lessDim until greaterDim - 1) +
                intArrayOf(diagonalEntries.shape[n - 1] + abs(realOffset)) +
                diagonalEntries.shape.slice(greaterDim - 1 until n - 1)
        val resTensor = zeros(resShape)

        for (i in 0 until diagonalEntries.asIntTensor().linearSize) {
            val multiIndex = diagonalEntries.asIntTensor().indices.index(i)

            var offset1 = 0
            var offset2 = abs(realOffset)
            if (realOffset < 0) {
                offset1 = offset2.also { offset2 = offset1 }
            }
            val diagonalMultiIndex = multiIndex.slice(0 until lessDim).toIntArray() +
                    intArrayOf(multiIndex[n - 1] + offset1) +
                    multiIndex.slice(lessDim until greaterDim - 1).toIntArray() +
                    intArrayOf(multiIndex[n - 1] + offset2) +
                    multiIndex.slice(greaterDim - 1 until n - 1).toIntArray()

            resTensor[diagonalMultiIndex] = diagonalEntries[multiIndex]
        }

        return resTensor.asIntTensor()
    }

    /**
     * Compares element-wise two int tensors
     *
     * @param other the tensor to compare with `input` tensor.
     * @param epsilon permissible error when comparing two Int values.
     * @return true if two tensors have the same shape and elements, false otherwise.
     */
    public fun Tensor<Int>.eq(other: Tensor<Int>): Boolean =
        asIntTensor().eq(other) { x, y -> x == y }

    private fun Tensor<Int>.eq(
        other: Tensor<Int>,
        eqFunction: (Int, Int) -> Boolean,
    ): Boolean {
        checkShapesCompatible(asIntTensor(), other)
        val n = asIntTensor().linearSize
        if (n != other.asIntTensor().linearSize) {
            return false
        }
        for (i in 0 until n) {
            if (!eqFunction(asIntTensor().source[i], other.asIntTensor().source[i])) {
                return false
            }
        }
        return true
    }

    /**
     * Concatenates a sequence of tensors with equal shapes along the first dimension.
     *
     * @param tensors the [List] of tensors with same shapes to concatenate
     * @return tensor with concatenation result
     */
    public fun stack(tensors: List<Tensor<Int>>): IntTensor {
        check(tensors.isNotEmpty()) { "List must have at least 1 element" }
        val shape = tensors[0].shape
        check(tensors.all { it.shape contentEquals shape }) { "Tensors must have same shapes" }
        val resShape = ShapeND(tensors.size) + shape
//        val resBuffer: List<Int> = tensors.flatMap {
//            it.asIntTensor().source.array.drop(it.asIntTensor().bufferStart)
//                .take(it.asIntTensor().linearSize)
//        }
        val resBuffer = tensors.map { it.asIntTensor().source }.concat()
        return IntTensor(resShape, resBuffer)
    }

    /**
     * Builds tensor from rows of the input tensor.
     *
     * @param indices the [IntArray] of 1-dimensional indices
     * @return tensor with rows corresponding to row by [indices]
     */
    public fun Tensor<Int>.rowsByIndices(indices: IntArray): IntTensor = stack(indices.map { getTensor(it) })

    private inline fun StructureND<Int>.foldDimToInt(
        dim: Int,
        keepDim: Boolean,
        foldFunction: (IntArray) -> Int,
    ): IntTensor {
        check(dim < dimension) { "Dimension $dim out of range $dimension" }
        val resShape = if (keepDim) {
            shape.first(dim) + intArrayOf(1) + shape.last(dimension - dim - 1)
        } else {
            shape.first(dim) + shape.last(dimension - dim - 1)
        }
        val resNumElements = resShape.linearSize
        val init = foldFunction(IntArray(1) { 0 })
        val resTensor = IntTensor(
            resShape,
            IntBuffer(resNumElements) { init }
        )
        for (index in resTensor.indices) {
            val prefix = index.take(dim).toIntArray()
            val suffix = index.takeLast(dimension - dim - 1).toIntArray()
            resTensor[index] = foldFunction(IntArray(shape[dim]) { i ->
                asIntTensor()[prefix + intArrayOf(i) + suffix]
            })
        }
        return resTensor
    }


    override fun StructureND<Int>.sum(): Int = reduceElements { it.array.sum() }

    override fun StructureND<Int>.sum(dim: Int, keepDim: Boolean): IntTensor =
        foldDimToInt(dim, keepDim) { x -> x.sum() }

    override fun StructureND<Int>.min(): Int = reduceElements { it.array.min() }

    override fun StructureND<Int>.min(dim: Int, keepDim: Boolean): IntTensor =
        foldDimToInt(dim, keepDim) { x -> x.minOrNull()!! }

    override fun StructureND<Int>.argMin(dim: Int, keepDim: Boolean): Tensor<Int> = foldDimToInt(dim, keepDim) { x ->
        x.withIndex().minBy { it.value }.index
    }

    override fun StructureND<Int>.max(): Int = reduceElements { it.array.max() }

    override fun StructureND<Int>.max(dim: Int, keepDim: Boolean): IntTensor =
        foldDimToInt(dim, keepDim) { x -> x.max() }


    override fun StructureND<Int>.argMax(dim: Int, keepDim: Boolean): IntTensor =
        foldDimToInt(dim, keepDim) { x ->
            x.withIndex().maxBy { it.value }.index
        }

    public fun StructureND<Int>.mean(): Double = sum().toDouble() / indices.linearSize
}

public val Int.Companion.tensorAlgebra: IntTensorAlgebra get() = IntTensorAlgebra
public val IntRing.tensorAlgebra: IntTensorAlgebra get() = IntTensorAlgebra


