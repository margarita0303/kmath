/*
 * Copyright 2018-2022 KMath contributors.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the license/LICENSE.txt file.
 */

@file:OptIn(UnstableKMathAPI::class)

package space.kscience.kmath.expressions

import space.kscience.kmath.UnstableKMathAPI
import space.kscience.kmath.operations.DoubleField
import kotlin.contracts.InvocationKind
import kotlin.contracts.contract
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFails

internal inline fun diff(
    order: Int,
    vararg parameters: Pair<Symbol, Double>,
    block: DSField<Double, DoubleField>.() -> Unit,
) {
    contract { callsInPlace(block, InvocationKind.EXACTLY_ONCE) }
    DSField(DoubleField, order, mapOf(*parameters)).block()
}

internal class DSTest {
    private val x by symbol
    private val y by symbol

    @Test
    fun dsAlgebraTest() {
        diff(2, x to 1.0, y to 1.0) {
            val x = bindSymbol(x)//by binding()
            val y = bindSymbol("y")
            val z = x * (-sin(x * y) + y) + 2.0
            println(z.derivative(x))
            println(z.derivative(y, x))
            assertEquals(z.derivative(x, y), z.derivative(y, x))
            // check improper order cause failure
            assertFails { z.derivative(x, x, y) }
        }
    }

    @Test
    fun dsExpressionTest() {
        val f = DSFieldExpression(DoubleField) {
            val x by binding
            val y by binding
            x.pow(2) + 2 * x * y + y.pow(2) + 1
        }

        assertEquals(10.0, f(x to 1.0, y to 2.0))
        assertEquals(6.0, f.derivative(x)(x to 1.0, y to 2.0))
        assertEquals(2.0, f.derivative(x, x)(x to 1.234, y to -2.0))
        assertEquals(2.0, f.derivative(x, y)(x to 1.0, y to 2.0))
    }
}
