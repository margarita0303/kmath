package space.kscience.kmath.chains

import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.map
import kotlinx.coroutines.flow.runningReduce
import kotlinx.coroutines.flow.scan
import space.kscience.kmath.operations.ScaleOperations
import space.kscience.kmath.operations.Space
import space.kscience.kmath.operations.SpaceOperations
import space.kscience.kmath.operations.invoke

public fun <T> Flow<T>.cumulativeSum(space: SpaceOperations<T>): Flow<T> =
    space { runningReduce { sum, element -> sum + element } }

@ExperimentalCoroutinesApi
public fun <T, S> Flow<T>.mean(algebra: S): Flow<T> where S : Space<T>, S : ScaleOperations<T> = algebra {
    data class Accumulator(var sum: T, var num: Int)

    scan(Accumulator(zero, 0)) { sum, element ->
        sum.apply {
            this.sum += element
            this.num += 1
        }
    }.map { it.sum / it.num }
}
