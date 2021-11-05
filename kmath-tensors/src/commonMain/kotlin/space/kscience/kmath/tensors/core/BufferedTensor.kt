/*
 * Copyright 2018-2021 KMath contributors.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE file.
 */

package space.kscience.kmath.tensors.core

import space.kscience.kmath.misc.PerformancePitfall
import space.kscience.kmath.nd.Strides
import space.kscience.kmath.structures.MutableBuffer
import space.kscience.kmath.tensors.api.Tensor

/**
 * Represents [Tensor] over a [MutableBuffer] intended to be used through [DoubleTensor] and [IntTensor]
 */
public open class BufferedTensor<T> internal constructor(
    override val shape: IntArray,
    @PublishedApi internal val mutableBuffer: MutableBuffer<T>,
    @PublishedApi internal val bufferStart: Int,
) : Tensor<T> {

    /**
     * Buffer strides based on [TensorLinearStructure] implementation
     */
    override val shapeIndices: Strides get() = TensorLinearStructure(shape)

    /**
     * Number of elements in tensor
     */
    public val numElements: Int
        get() = shapeIndices.linearSize

    override fun get(index: IntArray): T = mutableBuffer[bufferStart + shapeIndices.offset(index)]

    override fun set(index: IntArray, value: T) {
        mutableBuffer[bufferStart + shapeIndices.offset(index)] = value
    }

    @PerformancePitfall
    override fun elements(): Sequence<Pair<IntArray, T>> = shapeIndices.asSequence().map {
        it to get(it)
    }
}
