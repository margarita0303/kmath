package space.kscience.kmath.linear

import kotlin.test.Test
import kotlin.test.assertEquals

class RealLUSolverTest {

    @Test
    fun testInvertOne() {
        val matrix = LinearSpace.real.one(2, 2)
        val inverted = LinearSpace.real.inverseWithLup(matrix)
        assertEquals(matrix, inverted)
    }

    @Test
    fun testDecomposition() {
        LinearSpace.real.run {
            val matrix = square(
                3.0, 1.0,
                1.0, 3.0
            )

            val lup = lup(matrix)

            //Check determinant
            assertEquals(8.0, lup.determinant)

            assertEquals(lup.p dot matrix, lup.l dot lup.u)
        }
    }

    @Test
    fun testInvert() {
        val matrix = LinearSpace.real.square(
            3.0, 1.0,
            1.0, 3.0
        )

        val inverted = LinearSpace.real.inverseWithLup(matrix)

        val expected = LinearSpace.real.square(
            0.375, -0.125,
            -0.125, 0.375
        )

        assertEquals(expected, inverted)
    }
}
