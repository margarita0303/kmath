/*
 * Copyright 2018-2021 KMath contributors.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the license/LICENSE.txt file.
 */

package space.kscience.kmath.multik

import org.jetbrains.kotlinx.multik.api.Multik
import org.jetbrains.kotlinx.multik.api.ndarrayOf
import org.jetbrains.kotlinx.multik.ndarray.data.DataType
import space.kscience.kmath.operations.FloatField

public object MultikFloatAlgebra : MultikDivisionTensorAlgebra<Float, FloatField>() {
    override val elementAlgebra: FloatField get() = FloatField
    override val type: DataType get() = DataType.FloatDataType

    override fun scalar(value: Float): MultikTensor<Float> = Multik.ndarrayOf(value).wrap()
}


public val Float.Companion.multikAlgebra: MultikTensorAlgebra<Float, FloatField> get() = MultikFloatAlgebra
public val FloatField.multikAlgebra: MultikTensorAlgebra<Float, FloatField> get() = MultikFloatAlgebra