/*
 * Copyright 2018-2021 KMath contributors.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the license/LICENSE.txt file.
 */

package space.kscience.kmath.integration

import space.kscience.kmath.misc.UnstableKMathAPI
import kotlin.jvm.JvmInline
import kotlin.reflect.KClass

public class UnivariateIntegrand<T> internal constructor(
    private val features: Map<KClass<*>, IntegrandFeature>,
    public val function: (Double) -> T,
) : Integrand {

    @Suppress("UNCHECKED_CAST")
    override fun <T : IntegrandFeature> getFeature(type: KClass<T>): T? = features[type] as? T

    public operator fun <F : IntegrandFeature> plus(pair: Pair<KClass<out F>, F>): UnivariateIntegrand<T> =
        UnivariateIntegrand(features + pair, function)

    public operator fun <F : IntegrandFeature> plus(feature: F): UnivariateIntegrand<T> =
        plus(feature::class to feature)
}

@Suppress("FunctionName")
public fun <T : Any> UnivariateIntegrand(
    function: (Double) -> T,
    vararg features: IntegrandFeature,
): UnivariateIntegrand<T> = UnivariateIntegrand(features.associateBy { it::class }, function)

public typealias UnivariateIntegrator<T> = Integrator<UnivariateIntegrand<T>>

@JvmInline
public value class IntegrationRange(public val range: ClosedRange<Double>) : IntegrandFeature

/**
 * Value of the integrand if it is present or null
 */
public val <T : Any> UnivariateIntegrand<T>.valueOrNull: T? get() = getFeature<IntegrandValue<T>>()?.value

/**
 * Value of the integrand or error
 */
public val <T : Any> UnivariateIntegrand<T>.value: T get() = valueOrNull ?: error("No value in the integrand")

/**
 * A shortcut method to integrate a [function] with additional [features]. Range must be provided in features.
 * The [function] is placed in the end position to allow passing a lambda.
 */
@UnstableKMathAPI
public fun <T: Any> UnivariateIntegrator<T>.integrate(
    vararg features: IntegrandFeature,
    function: (Double) -> T,
): UnivariateIntegrand<T> = integrate(UnivariateIntegrand(function, *features))

/**
 * A shortcut method to integrate a [function] in [range] with additional [features].
 * The [function] is placed in the end position to allow passing a lambda.
 */
@UnstableKMathAPI
public fun <T: Any> UnivariateIntegrator<T>.integrate(
    range: ClosedRange<Double>,
    vararg features: IntegrandFeature,
    function: (Double) -> T,
): UnivariateIntegrand<T> = integrate(UnivariateIntegrand(function, IntegrationRange(range), *features))

/**
 * A shortcut method to integrate a [function] in [range] with additional [features].
 * The [function] is placed in the end position to allow passing a lambda.
 */
@UnstableKMathAPI
public fun <T: Any> UnivariateIntegrator<T>.integrate(
    range: ClosedRange<Double>,
    featureBuilder: MutableList<IntegrandFeature>.() -> Unit = {},
    function: (Double) -> T,
): UnivariateIntegrand<T> {
    //TODO use dedicated feature builder class instead or add extensions to MutableList<IntegrandFeature>
    val features = buildList {
        featureBuilder()
        add(IntegrationRange(range))
    }
    return integrate(UnivariateIntegrand(function, *features.toTypedArray()))
}
