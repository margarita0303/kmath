package kscience.kmath.linear

import kscience.kmath.structures.Buffer
import kscience.kmath.structures.BufferFactory
import kscience.kmath.structures.Structure2D
import kscience.kmath.structures.asBuffer

public class MatrixBuilder(public val rows: Int, public val columns: Int) {
    public operator fun <T : Any> invoke(vararg elements: T): FeaturedMatrix<T> {
        require(rows * columns == elements.size) { "The number of elements ${elements.size} is not equal $rows * $columns" }
        val buffer = elements.asBuffer()
        return BufferMatrix(rows, columns, buffer)
    }

    //TODO add specific matrix builder functions like diagonal, etc
}

public fun Structure2D.Companion.build(rows: Int, columns: Int): MatrixBuilder = MatrixBuilder(rows, columns)

public fun <T : Any> Structure2D.Companion.row(vararg values: T): FeaturedMatrix<T> {
    val buffer = values.asBuffer()
    return BufferMatrix(1, values.size, buffer)
}

public inline fun <reified T : Any> Structure2D.Companion.row(
    size: Int,
    factory: BufferFactory<T> = Buffer.Companion::auto,
    noinline builder: (Int) -> T
): FeaturedMatrix<T> {
    val buffer = factory(size, builder)
    return BufferMatrix(1, size, buffer)
}

public fun <T : Any> Structure2D.Companion.column(vararg values: T): FeaturedMatrix<T> {
    val buffer = values.asBuffer()
    return BufferMatrix(values.size, 1, buffer)
}

public inline fun <reified T : Any> Structure2D.Companion.column(
    size: Int,
    factory: BufferFactory<T> = Buffer.Companion::auto,
    noinline builder: (Int) -> T
): FeaturedMatrix<T> {
    val buffer = factory(size, builder)
    return BufferMatrix(size, 1, buffer)
}