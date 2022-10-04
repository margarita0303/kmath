/*
 * Copyright 2018-2022 KMath contributors.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the license/LICENSE.txt file.
 */

package space.kscience.kmath.nd

import kotlin.math.max
import kotlin.native.concurrent.ThreadLocal

/**
 * A converter from linear index to multivariate index
 */
public interface ShapeIndexer : Iterable<IntArray> {
    public val shape: Shape

    /**
     * Get linear index from multidimensional index
     */
    public fun offset(index: IntArray): Int

    /**
     * Get multidimensional from linear
     */
    public fun index(offset: Int): IntArray

    /**
     * The size of linear buffer to accommodate all elements of ND-structure corresponding to strides
     */
    public val linearSize: Int

    // TODO introduce a fast way to calculate index of the next element?

    /**
     * Iterate over ND indices in a natural order
     */
    public fun asSequence(): Sequence<IntArray>

    override fun iterator(): Iterator<IntArray> = asSequence().iterator()

    override fun equals(other: Any?): Boolean
    override fun hashCode(): Int
}

/**
 * Linear transformation of indexes
 */
public abstract class Strides : ShapeIndexer {
    /**
     * Array strides
     */
    public abstract val strides: IntArray

    public override fun offset(index: IntArray): Int = index.mapIndexed { i, value ->
        if (value < 0 || value >= shape[i]) throw IndexOutOfBoundsException("Index $value out of shape bounds: (0,${this.shape[i]})")
        value * strides[i]
    }.sum()

    // TODO introduce a fast way to calculate index of the next element?

    /**
     * Iterate over ND indices in a natural order
     */
    public override fun asSequence(): Sequence<IntArray> = (0 until linearSize).asSequence().map(::index)

    public companion object{
        public fun linearSizeOf(shape: IntArray): Int = shape.reduce(Int::times)
    }
}

/**
 * Column-first [Strides]. Columns are represented as continuous arrays
 */
public class ColumnStrides(override val shape: IntArray) : Strides() {
    override val linearSize: Int get() = strides[shape.size]

    /**
     * Strides for memory access
     */
    override val strides: IntArray by lazy {
        sequence {
            var current = 1
            yield(1)

            shape.forEach {
                current *= it
                yield(current)
            }
        }.toList().toIntArray()
    }

    override fun index(offset: Int): IntArray {
        val res = IntArray(shape.size)
        var current = offset
        var strideIndex = strides.size - 2

        while (strideIndex >= 0) {
            res[strideIndex] = (current / strides[strideIndex])
            current %= strides[strideIndex]
            strideIndex--
        }

        return res
    }

    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is ColumnStrides) return false
        return shape.contentEquals(other.shape)
    }

    override fun hashCode(): Int = shape.contentHashCode()


    public companion object
}

/**
 * This [Strides] implementation follows the last dimension first convention
 * For more information: https://numpy.org/doc/stable/reference/generated/numpy.ndarray.strides.html
 *
 * @param shape the shape of the tensor.
 */
public class RowStrides(override val shape: IntArray) : Strides() {

    override val strides: IntArray by lazy {
        val nDim = shape.size
        val res = IntArray(nDim)
        if (nDim == 0) return@lazy res

        var current = nDim - 1
        res[current] = 1

        while (current > 0) {
            res[current - 1] = max(1, shape[current]) * res[current]
            current--
        }
        res
    }

    override fun index(offset: Int): IntArray {
        val res = IntArray(shape.size)
        var current = offset
        var strideIndex = 0

        while (strideIndex < shape.size) {
            res[strideIndex] = (current / strides[strideIndex])
            current %= strides[strideIndex]
            strideIndex++
        }
        return res
    }

    override val linearSize: Int get() = linearSizeOf(shape)

    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is RowStrides) return false
        return shape.contentEquals(other.shape)
    }

    override fun hashCode(): Int = shape.contentHashCode()

    public companion object

}

@ThreadLocal
private val defaultStridesCache = HashMap<IntArray, Strides>()

/**
 * Cached builder for default strides
 */
public fun Strides(shape: IntArray): Strides = defaultStridesCache.getOrPut(shape) { RowStrides(shape) }