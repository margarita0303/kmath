/*
 * Copyright 2018-2021 KMath contributors.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the license/LICENSE.txt file.
 */

package space.kscience.kmath.functions

import space.kscience.kmath.operations.*
import kotlin.js.JsName
import kotlin.jvm.JvmName


/**
 * Abstraction of rational function.
 */
public interface RationalFunction<C, P: Polynomial<C>> {
    public val numerator: P
    public val denominator: P
    public operator fun component1(): P = numerator
    public operator fun component2(): P = denominator
}

/**
 * Abstraction of field of rational functions of type [R] with respect to polynomials of type [P] and constants of type
 * [C].
 *
 * @param C the type of constants. Polynomials have them as coefficients in their terms.
 * @param P the type of polynomials. Rational functions have them as numerators and denominators in them.
 * @param R the type of rational functions.
 */
@Suppress("INAPPLICABLE_JVM_NAME", "PARAMETER_NAME_CHANGED_ON_OVERRIDE") // FIXME: Waiting for KT-31420
public interface RationalFunctionalSpace<C, P: Polynomial<C>, R: RationalFunction<C, P>> : Ring<R> {
    /**
     * Returns sum of the constant and the integer represented as constant (member of underlying ring).
     *
     * The operation is equivalent to adding [other] copies of unit of underlying ring to [this].
     */
    public operator fun C.plus(other: Int): C
    /**
     * Returns difference between the constant and the integer represented as constant (member of underlying ring).
     *
     * The operation is equivalent to subtraction [other] copies of unit of underlying ring from [this].
     */
    public operator fun C.minus(other: Int): C
    /**
     * Returns product of the constant and the integer represented as constant (member of underlying ring).
     *
     * The operation is equivalent to sum of [other] copies of [this].
     */
    public operator fun C.times(other: Int): C

    /**
     * Returns sum of the integer represented as constant (member of underlying ring) and the constant.
     *
     * The operation is equivalent to adding [this] copies of unit of underlying ring to [other].
     */
    public operator fun Int.plus(other: C): C
    /**
     * Returns difference between the integer represented as constant (member of underlying ring) and the constant.
     *
     * The operation is equivalent to subtraction [this] copies of unit of underlying ring from [other].
     */
    public operator fun Int.minus(other: C): C
    /**
     * Returns product of the integer represented as constant (member of underlying ring) and the constant.
     *
     * The operation is equivalent to sum of [this] copies of [other].
     */
    public operator fun Int.times(other: C): C

    /**
     * Converts the integer [value] to constant.
     */
    public fun constantNumber(value: Int): C = constantOne * value
    /**
     * Converts the integer to constant.
     */
    public fun Int.asConstant(): C = constantNumber(this)

    /**
     * Returns sum of the constant and the integer represented as polynomial.
     *
     * The operation is equivalent to adding [other] copies of unit polynomial to [this].
     */
    public operator fun P.plus(other: Int): P
    /**
     * Returns difference between the constant and the integer represented as polynomial.
     *
     * The operation is equivalent to subtraction [other] copies of unit polynomial from [this].
     */
    public operator fun P.minus(other: Int): P
    /**
     * Returns product of the constant and the integer represented as polynomial.
     *
     * The operation is equivalent to sum of [other] copies of [this].
     */
    public operator fun P.times(other: Int): P

    /**
     * Returns sum of the integer represented as polynomial and the constant.
     *
     * The operation is equivalent to adding [this] copies of unit polynomial to [other].
     */
    public operator fun Int.plus(other: P): P
    /**
     * Returns difference between the integer represented as polynomial and the constant.
     *
     * The operation is equivalent to subtraction [this] copies of unit polynomial from [other].
     */
    public operator fun Int.minus(other: P): P
    /**
     * Returns product of the integer represented as polynomial and the constant.
     *
     * The operation is equivalent to sum of [this] copies of [other].
     */
    public operator fun Int.times(other: P): P

    /**
     * Converts the integer [value] to polynomial.
     */
    public fun polynomialNumber(value: Int): P = polynomialOne * value
    /**
     * Converts the integer to polynomial.
     */
    public fun Int.asPolynomial(): P = polynomialNumber(this)

    /**
     * Returns sum of the rational function and the integer represented as rational function.
     *
     * The operation is equivalent to adding [other] copies of unit polynomial to [this].
     */
    public operator fun R.plus(other: Int): R = addMultipliedByDoubling(this, one, other)
    /**
     * Returns difference between the rational function and the integer represented as rational function.
     *
     * The operation is equivalent to subtraction [other] copies of unit polynomial from [this].
     */
    public operator fun R.minus(other: Int): R = addMultipliedByDoubling(this, one, -other)
    /**
     * Returns product of the rational function and the integer represented as rational function.
     *
     * The operation is equivalent to sum of [other] copies of [this].
     */
    public operator fun R.times(other: Int): R = multiplyByDoubling(this, other)
    /**
     * Returns quotient of the rational function and the integer represented as rational function.
     *
     * The operation is equivalent to creating a new rational function by preserving numerator of [this] and
     * multiplication denominator of [this] to [other].
     */
    public operator fun R.div(other: Int): R = this / multiplyByDoubling(one, other)

    /**
     * Returns sum of the integer represented as rational function and the rational function.
     *
     * The operation is equivalent to adding [this] copies of unit polynomial to [other].
     */
    public operator fun Int.plus(other: R): R = addMultipliedByDoubling(other, one, this)
    /**
     * Returns difference between the integer represented as rational function and the rational function.
     *
     * The operation is equivalent to subtraction [this] copies of unit polynomial from [other].
     */
    public operator fun Int.minus(other: R): R = addMultipliedByDoubling(-other, one, this)
    /**
     * Returns product of the integer represented as rational function and the rational function.
     *
     * The operation is equivalent to sum of [this] copies of [other].
     */
    public operator fun Int.times(other: R): R = multiplyByDoubling(other, this)
    /**
     * Returns quotient of the integer represented as rational function and the rational function.
     *
     * The operation is equivalent to creating a new rational function which numerator is [this] times denominator of
     * [other] and which denominator is [other]'s numerator.
     */
    public operator fun Int.div(other: R): R = multiplyByDoubling(one / other, this)

    /**
     * Converts the integer [value] to rational function.
     */
    public fun number(value: Int): R = one * value
    /**
     * Converts the integer to rational function.
     */
    public fun Int.asRationalFunction(): R = number(this)

    /**
     * Returns the same constant.
     */
    @JvmName("unaryPlusConstant")
    @JsName("unaryPlusConstant")
    public operator fun C.unaryPlus(): C = this
    /**
     * Returns negation of the constant.
     */
    @JvmName("unaryMinusConstant")
    @JsName("unaryMinusConstant")
    public operator fun C.unaryMinus(): C
    /**
     * Returns sum of the constants.
     */
    @JvmName("plusConstantConstant")
    @JsName("plusConstantConstant")
    public operator fun C.plus(other: C): C
    /**
     * Returns difference of the constants.
     */
    @JvmName("minusConstantConstant")
    @JsName("minusConstantConstant")
    public operator fun C.minus(other: C): C
    /**
     * Returns product of the constants.
     */
    @JvmName("timesConstantConstant")
    @JsName("timesConstantConstant")
    public operator fun C.times(other: C): C
    /**
     * Raises [arg] to the integer power [exponent].
     */
    @JvmName("powerConstant")
    @JsName("powerConstant")
    public fun power(arg: C, exponent: UInt) : C

    /**
     * Instance of zero constant (zero of the underlying ring).
     */
    public val constantZero: C
    /**
     * Instance of unit constant (unit of the underlying ring).
     */
    public val constantOne: C

    /**
     * Returns sum of the constant represented as polynomial and the polynomial.
     */
    public operator fun C.plus(other: P): P
    /**
     * Returns difference between the constant represented as polynomial and the polynomial.
     */
    public operator fun C.minus(other: P): P
    /**
     * Returns product of the constant represented as polynomial and the polynomial.
     */
    public operator fun C.times(other: P): P

    /**
     * Returns sum of the constant represented as polynomial and the polynomial.
     */
    public operator fun P.plus(other: C): P
    /**
     * Returns difference between the constant represented as polynomial and the polynomial.
     */
    public operator fun P.minus(other: C): P
    /**
     * Returns product of the constant represented as polynomial and the polynomial.
     */
    public operator fun P.times(other: C): P

    /**
     * Converts the constant [value] to polynomial.
     */
    public fun polynomialNumber(value: C): P = polynomialOne * value
    /**
     * Converts the constant to polynomial.
     */
    public fun C.asPolynomial(): P = polynomialNumber(this)

    /**
     * Returns the same polynomial.
     */
    public operator fun P.unaryPlus(): P = this
    /**
     * Returns negation of the polynomial.
     */
    public operator fun P.unaryMinus(): P
    /**
     * Returns sum of the polynomials.
     */
    public operator fun P.plus(other: P): P
    /**
     * Returns difference of the polynomials.
     */
    public operator fun P.minus(other: P): P
    /**
     * Returns product of the polynomials.
     */
    public operator fun P.times(other: P): P
    /**
     * Returns quotient of the polynomials as rational function.
     */
    public operator fun P.div(other: P): R
    /**
     * Raises [arg] to the integer power [exponent].
     */
    public fun power(arg: P, exponent: UInt) : P

    /**
     * Instance of zero polynomial (zero of the polynomial ring).
     */
    public val polynomialZero: P
    /**
     * Instance of unit polynomial (unit of the polynomial ring).
     */
    public val polynomialOne: P

    /**
     * Returns sum of the constant represented as rational function and the rational function.
     */
    public operator fun C.plus(other: R): R
    /**
     * Returns difference between the constant represented as polynomial and the rational function.
     */
    public operator fun C.minus(other: R): R
    /**
     * Returns product of the constant represented as polynomial and the rational function.
     */
    public operator fun C.times(other: R): R
    /**
     * Returns quotient of the constant represented as polynomial and the rational function.
     */
    public operator fun C.div(other: R): R

    /**
     * Returns sum of the rational function and the constant represented as rational function.
     */
    public operator fun R.plus(other: C): R
    /**
     * Returns difference between the rational function and the constant represented as rational function.
     */
    public operator fun R.minus(other: C): R
    /**
     * Returns product of the rational function and the constant represented as rational function.
     */
    public operator fun R.times(other: C): R
    /**
     * Returns quotient of the rational function and the constant represented as rational function.
     */
    public operator fun R.div(other: C): R

    /**
     * Converts the constant [value] to rational function.
     */
    public fun number(value: C): R = one * value
    /**
     * Converts the constant to rational function.
     */
    public fun C.asRationalFunction(): R = number(this)

    /**
     * Returns sum of the polynomial represented as rational function and the rational function.
     */
    public operator fun P.plus(other: R): R
    /**
     * Returns difference between the polynomial represented as polynomial and the rational function.
     */
    public operator fun P.minus(other: R): R
    /**
     * Returns product of the polynomial represented as polynomial and the rational function.
     */
    public operator fun P.times(other: R): R
    /**
     * Returns quotient of the polynomial represented as polynomial and the rational function.
     */
    public operator fun P.div(other: R): R

    /**
     * Returns sum of the rational function and the polynomial represented as rational function.
     */
    public operator fun R.plus(other: P): R
    /**
     * Returns difference between the rational function and the polynomial represented as rational function.
     */
    public operator fun R.minus(other: P): R
    /**
     * Returns product of the rational function and the polynomial represented as rational function.
     */
    public operator fun R.times(other: P): R
    /**
     * Returns quotient of the rational function and the polynomial represented as rational function.
     */
    public operator fun R.div(other: P): R

    /**
     * Converts the polynomial [value] to rational function.
     */
    public fun number(value: P): R = one * value
    /**
     * Converts the polynomial to rational function.
     */
    public fun P.asRationalFunction(): R = number(this)

    /**
     * Returns the same rational function.
     */
    public override operator fun R.unaryPlus(): R = this
    /**
     * Returns negation of the rational function.
     */
    public override operator fun R.unaryMinus(): R
    /**
     * Returns sum of the rational functions.
     */
    public override operator fun R.plus(other: R): R
    /**
     * Returns difference of the rational functions.
     */
    public override operator fun R.minus(other: R): R
    /**
     * Returns product of the rational functions.
     */
    public override operator fun R.times(other: R): R
    /**
     * Returns quotient of the rational functions.
     */
    public operator fun R.div(other: R): R
    /**
     * Raises [arg] to the integer power [exponent].
     */
    public override fun power(arg: R, exponent: UInt) : R = exponentiateBySquaring(arg, exponent)

    /**
     * Instance of zero rational function (zero of the rational functions ring).
     */
    public override val zero: R
    /**
     * Instance of unit polynomial (unit of the rational functions ring).
     */
    public override val one: R

    /**
     * Degree of the polynomial, [see also](https://en.wikipedia.org/wiki/Degree_of_a_polynomial). If the polynomial is
     * zero, degree is -1.
     */
    public val P.degree: Int

    /**
     * Degree of the polynomial, [see also](https://en.wikipedia.org/wiki/Degree_of_a_polynomial). If the polynomial is
     * zero, degree is -1.
     */
    public val R.numeratorDegree: Int get() = numerator.degree
    /**
     * Degree of the polynomial, [see also](https://en.wikipedia.org/wiki/Degree_of_a_polynomial). If the polynomial is
     * zero, degree is -1.
     */
    public val R.denominatorDegree: Int get() = denominator.degree

    override fun add(left: R, right: R): R = left + right
    override fun multiply(left: R, right: R): R = left * right
}

/**
 * Abstraction of field of rational functions of type [R] with respect to polynomials of type [P] and constants of type
 * [C]. It also assumes that there is provided [ring] (of type [A]), that provides constant-wise operations.
 *
 * @param C the type of constants. Polynomials have them as coefficients in their terms.
 * @param P the type of polynomials. Rational functions have them as numerators and denominators in them.
 * @param R the type of rational functions.
 * @param A the type of algebraic structure (precisely, of ring) provided for constants.
 */
@Suppress("INAPPLICABLE_JVM_NAME") // FIXME: Waiting for KT-31420
public interface RationalFunctionalSpaceOverRing<C, P: Polynomial<C>, R: RationalFunction<C, P>, A: Ring<C>> : RationalFunctionalSpace<C, P, R> {

    public val ring: A

    /**
     * Returns sum of the constant and the integer represented as constant (member of underlying ring).
     *
     * The operation is equivalent to adding [other] copies of unit of underlying ring to [this].
     */
    public override operator fun C.plus(other: Int): C = ring { addMultipliedByDoubling(this@plus, one, other) }
    /**
     * Returns difference between the constant and the integer represented as constant (member of underlying ring).
     *
     * The operation is equivalent to subtraction [other] copies of unit of underlying ring from [this].
     */
    public override operator fun C.minus(other: Int): C = ring { addMultipliedByDoubling(this@minus, one, -other) }
    /**
     * Returns product of the constant and the integer represented as constant (member of underlying ring).
     *
     * The operation is equivalent to sum of [other] copies of [this].
     */
    public override operator fun C.times(other: Int): C = ring { multiplyByDoubling(this@times, other) }

    /**
     * Returns sum of the integer represented as constant (member of underlying ring) and the constant.
     *
     * The operation is equivalent to adding [this] copies of unit of underlying ring to [other].
     */
    public override operator fun Int.plus(other: C): C = ring { addMultipliedByDoubling(other, one, this@plus) }
    /**
     * Returns difference between the integer represented as constant (member of underlying ring) and the constant.
     *
     * The operation is equivalent to subtraction [this] copies of unit of underlying ring from [other].
     */
    public override operator fun Int.minus(other: C): C = ring { addMultipliedByDoubling(-other, one, this@minus) }
    /**
     * Returns product of the integer represented as constant (member of underlying ring) and the constant.
     *
     * The operation is equivalent to sum of [this] copies of [other].
     */
    public override operator fun Int.times(other: C): C = ring { multiplyByDoubling(other, this@times) }

    /**
     * Returns the same constant.
     */
    @JvmName("unaryPlusConstant")
    public override operator fun C.unaryPlus(): C = ring { +this@unaryPlus }
    /**
     * Returns negation of the constant.
     */
    @JvmName("unaryMinusConstant")
    public override operator fun C.unaryMinus(): C = ring { -this@unaryMinus }
    /**
     * Returns sum of the constants.
     */
    @JvmName("plusConstantConstant")
    public override operator fun C.plus(other: C): C = ring { this@plus + other }
    /**
     * Returns difference of the constants.
     */
    @JvmName("minusConstantConstant")
    public override operator fun C.minus(other: C): C = ring { this@minus - other }
    /**
     * Returns product of the constants.
     */
    @JvmName("timesConstantConstant")
    public override operator fun C.times(other: C): C = ring { this@times * other }
    /**
     * Raises [arg] to the integer power [exponent].
     */
    @JvmName("powerConstant")
    public override fun power(arg: C, exponent: UInt) : C = ring { power(arg, exponent) }

    /**
     * Instance of zero constant (zero of the underlying ring).
     */
    public override val constantZero: C get() = ring.zero
    /**
     * Instance of unit constant (unit of the underlying ring).
     */
    public override val constantOne: C get() = ring.one
}

/**
 * Abstraction of field of rational functions of type [R] with respect to polynomials of type [P] and constants of type
 * [C]. It also assumes that there is provided [polynomialRing] (of type [AP]), that provides constant- and
 * polynomial-wise operations.
 *
 * @param C the type of constants. Polynomials have them as coefficients in their terms.
 * @param P the type of polynomials. Rational functions have them as numerators and denominators in them.
 * @param R the type of rational functions.
 * @param AP the type of algebraic structure (precisely, of ring) provided for polynomials.
 */
@Suppress("INAPPLICABLE_JVM_NAME") // FIXME: Waiting for KT-31420
public interface RationalFunctionalSpaceOverPolynomialSpace<
        C,
        P: Polynomial<C>,
        R: RationalFunction<C, P>,
        AP: PolynomialSpace<C, P>,
        > : RationalFunctionalSpace<C, P, R> {

    public val polynomialRing: AP

    /**
     * Returns sum of the constant and the integer represented as constant (member of underlying ring).
     *
     * The operation is equivalent to adding [other] copies of unit of underlying ring to [this].
     */
    public override operator fun C.plus(other: Int): C = polynomialRing { this@plus + other }
    /**
     * Returns difference between the constant and the integer represented as constant (member of underlying ring).
     *
     * The operation is equivalent to subtraction [other] copies of unit of underlying ring from [this].
     */
    public override operator fun C.minus(other: Int): C = polynomialRing { this@minus - other }
    /**
     * Returns product of the constant and the integer represented as constant (member of underlying ring).
     *
     * The operation is equivalent to sum of [other] copies of [this].
     */
    public override operator fun C.times(other: Int): C = polynomialRing { this@times * other }

    /**
     * Returns sum of the integer represented as constant (member of underlying ring) and the constant.
     *
     * The operation is equivalent to adding [this] copies of unit of underlying ring to [other].
     */
    public override operator fun Int.plus(other: C): C = polynomialRing { this@plus + other }
    /**
     * Returns difference between the integer represented as constant (member of underlying ring) and the constant.
     *
     * The operation is equivalent to subtraction [this] copies of unit of underlying ring from [other].
     */
    public override operator fun Int.minus(other: C): C = polynomialRing { this@minus - other }
    /**
     * Returns product of the integer represented as constant (member of underlying ring) and the constant.
     *
     * The operation is equivalent to sum of [this] copies of [other].
     */
    public override operator fun Int.times(other: C): C = polynomialRing { this@times * other }

    /**
     * Converts the integer [value] to constant.
     */
    public override fun constantNumber(value: Int): C = polynomialRing { constantNumber(value) }
    /**
     * Converts the integer to constant.
     */
    override fun Int.asConstant(): C = polynomialRing { asConstant() }

    /**
     * Returns sum of the constant and the integer represented as polynomial.
     *
     * The operation is equivalent to adding [other] copies of unit polynomial to [this].
     */
    public override operator fun P.plus(other: Int): P = polynomialRing { this@plus + other }
    /**
     * Returns difference between the constant and the integer represented as polynomial.
     *
     * The operation is equivalent to subtraction [other] copies of unit polynomial from [this].
     */
    public override operator fun P.minus(other: Int): P = polynomialRing { this@minus - other }
    /**
     * Returns product of the constant and the integer represented as polynomial.
     *
     * The operation is equivalent to sum of [other] copies of [this].
     */
    public override operator fun P.times(other: Int): P = polynomialRing { this@times * other }

    /**
     * Returns sum of the integer represented as polynomial and the constant.
     *
     * The operation is equivalent to adding [this] copies of unit polynomial to [other].
     */
    public override operator fun Int.plus(other: P): P = polynomialRing { this@plus + other }
    /**
     * Returns difference between the integer represented as polynomial and the constant.
     *
     * The operation is equivalent to subtraction [this] copies of unit polynomial from [other].
     */
    public override operator fun Int.minus(other: P): P = polynomialRing { this@minus - other }
    /**
     * Returns product of the integer represented as polynomial and the constant.
     *
     * The operation is equivalent to sum of [this] copies of [other].
     */
    public override operator fun Int.times(other: P): P = polynomialRing { this@times * other }

    /**
     * Converts the integer [value] to polynomial.
     */
    public override fun polynomialNumber(value: Int): P = polynomialRing { number(value) }
    /**
     * Converts the integer to polynomial.
     */
    public override fun Int.asPolynomial(): P = polynomialRing { asPolynomial() }

    /**
     * Returns the same constant.
     */
    @JvmName("unaryPlusConstant")
    public override operator fun C.unaryPlus(): C = polynomialRing { +this@unaryPlus }
    /**
     * Returns negation of the constant.
     */
    @JvmName("unaryMinusConstant")
    public override operator fun C.unaryMinus(): C = polynomialRing { -this@unaryMinus }
    /**
     * Returns sum of the constants.
     */
    @JvmName("plusConstantConstant")
    public override operator fun C.plus(other: C): C = polynomialRing { this@plus + other }
    /**
     * Returns difference of the constants.
     */
    @JvmName("minusConstantConstant")
    public override operator fun C.minus(other: C): C = polynomialRing { this@minus - other }
    /**
     * Returns product of the constants.
     */
    @JvmName("timesConstantConstant")
    public override operator fun C.times(other: C): C = polynomialRing { this@times * other }
    /**
     * Raises [arg] to the integer power [exponent].
     */
    @JvmName("powerConstant")
    public override fun power(arg: C, exponent: UInt) : C = polynomialRing { power(arg, exponent) }

    /**
     * Instance of zero constant (zero of the underlying ring).
     */
    public override val constantZero: C get() = polynomialRing.constantZero
    /**
     * Instance of unit constant (unit of the underlying ring).
     */
    public override val constantOne: C get() = polynomialRing.constantOne

    /**
     * Returns sum of the constant represented as polynomial and the polynomial.
     */
    public override operator fun C.plus(other: P): P = polynomialRing { this@plus + other }
    /**
     * Returns difference between the constant represented as polynomial and the polynomial.
     */
    public override operator fun C.minus(other: P): P = polynomialRing { this@minus - other }
    /**
     * Returns product of the constant represented as polynomial and the polynomial.
     */
    public override operator fun C.times(other: P): P = polynomialRing { this@times * other }

    /**
     * Returns sum of the constant represented as polynomial and the polynomial.
     */
    public override operator fun P.plus(other: C): P = polynomialRing { this@plus + other }
    /**
     * Returns difference between the constant represented as polynomial and the polynomial.
     */
    public override operator fun P.minus(other: C): P = polynomialRing { this@minus - other }
    /**
     * Returns product of the constant represented as polynomial and the polynomial.
     */
    public override operator fun P.times(other: C): P = polynomialRing { this@times * other }

    /**
     * Converts the constant [value] to polynomial.
     */
    public override fun polynomialNumber(value: C): P = polynomialRing { number(value) }
    /**
     * Converts the constant to polynomial.
     */
    public override fun C.asPolynomial(): P = polynomialRing { asPolynomial() }

    /**
     * Returns the same polynomial.
     */
    public override operator fun P.unaryPlus(): P = polynomialRing { +this@unaryPlus }
    /**
     * Returns negation of the polynomial.
     */
    public override operator fun P.unaryMinus(): P = polynomialRing { -this@unaryMinus }
    /**
     * Returns sum of the polynomials.
     */
    public override operator fun P.plus(other: P): P = polynomialRing { this@plus + other }
    /**
     * Returns difference of the polynomials.
     */
    public override operator fun P.minus(other: P): P = polynomialRing { this@minus - other }
    /**
     * Returns product of the polynomials.
     */
    public override operator fun P.times(other: P): P = polynomialRing { this@times * other }
    /**
     * Raises [arg] to the integer power [exponent].
     */
    public override fun power(arg: P, exponent: UInt) : P = polynomialRing { power(arg, exponent) }

    /**
     * Instance of zero polynomial (zero of the polynomial ring).
     */
    public override val polynomialZero: P get() = polynomialRing.zero
    /**
     * Instance of unit polynomial (unit of the polynomial ring).
     */
    public override val polynomialOne: P get() = polynomialRing.one

    /**
     * Degree of the polynomial, [see also](https://en.wikipedia.org/wiki/Degree_of_a_polynomial). If the polynomial is
     * zero, degree is -1.
     */
    public override val P.degree: Int get() = polynomialRing { this@degree.degree }
}

/**
 * Abstraction of field of rational functions of type [R] with respect to polynomials of type [P] and constants of type
 * [C]. It also assumes that there is provided constructor
 *
 * @param C the type of constants. Polynomials have them as coefficients in their terms.
 * @param P the type of polynomials. Rational functions have them as numerators and denominators in them.
 * @param R the type of rational functions.
 */
@Suppress("INAPPLICABLE_JVM_NAME") // FIXME: Waiting for KT-31420
public abstract class PolynomialSpaceOfFractions<
        C,
        P: Polynomial<C>,
        R: RationalFunction<C, P>,
        > : RationalFunctionalSpace<C, P, R> {
    protected abstract fun constructRationalFunction(numerator: P, denominator: P = polynomialOne) : R

    /**
     * Returns sum of the rational function and the integer represented as rational function.
     *
     * The operation is equivalent to adding [other] copies of unit polynomial to [this].
     */
    public override operator fun R.plus(other: Int): R =
        constructRationalFunction(
            numerator + denominator * other,
            denominator
        )
    /**
     * Returns difference between the rational function and the integer represented as rational function.
     *
     * The operation is equivalent to subtraction [other] copies of unit polynomial from [this].
     */
    public override operator fun R.minus(other: Int): R =
        constructRationalFunction(
            numerator - denominator * other,
            denominator
        )
    /**
     * Returns product of the rational function and the integer represented as rational function.
     *
     * The operation is equivalent to sum of [other] copies of [this].
     */
    public override operator fun R.times(other: Int): R =
        constructRationalFunction(
            numerator * other,
            denominator
        )

    public override operator fun R.div(other: Int): R =
        constructRationalFunction(
            numerator,
            denominator * other
        )

    /**
     * Returns sum of the integer represented as rational function and the rational function.
     *
     * The operation is equivalent to adding [this] copies of unit polynomial to [other].
     */
    public override operator fun Int.plus(other: R): R =
        constructRationalFunction(
            other.denominator * this + other.numerator,
            other.denominator
        )
    /**
     * Returns difference between the integer represented as rational function and the rational function.
     *
     * The operation is equivalent to subtraction [this] copies of unit polynomial from [other].
     */
    public override operator fun Int.minus(other: R): R =
        constructRationalFunction(
            other.denominator * this - other.numerator,
            other.denominator
        )
    /**
     * Returns product of the integer represented as rational function and the rational function.
     *
     * The operation is equivalent to sum of [this] copies of [other].
     */
    public override operator fun Int.times(other: R): R =
        constructRationalFunction(
            this * other.numerator,
            other.denominator
        )

    public override operator fun Int.div(other: R): R =
        constructRationalFunction(
            this * other.denominator,
            other.numerator
        )

    /**
     * Converts the integer [value] to rational function.
     */
    public override fun number(value: Int): R = constructRationalFunction(polynomialNumber(value))

    /**
     * Returns quotient of the polynomials as rational function.
     */
    public override operator fun P.div(other: P): R = constructRationalFunction(this, other)

    /**
     * Returns sum of the constant represented as rational function and the rational function.
     */
    public override operator fun C.plus(other: R): R =
        constructRationalFunction(
            other.denominator * this + other.numerator,
            other.denominator
        )
    /**
     * Returns difference between the constant represented as polynomial and the rational function.
     */
    public override operator fun C.minus(other: R): R =
        constructRationalFunction(
            other.denominator * this - other.numerator,
            other.denominator
        )
    /**
     * Returns product of the constant represented as polynomial and the rational function.
     */
    public override operator fun C.times(other: R): R =
        constructRationalFunction(
            this * other.numerator,
            other.denominator
        )

    public override operator fun C.div(other: R): R =
        constructRationalFunction(
            this * other.denominator,
            other.numerator
        )

    /**
     * Returns sum of the constant represented as rational function and the rational function.
     */
    public override operator fun R.plus(other: C): R =
        constructRationalFunction(
            numerator + denominator * other,
            denominator
        )
    /**
     * Returns difference between the constant represented as rational function and the rational function.
     */
    public override operator fun R.minus(other: C): R =
        constructRationalFunction(
            numerator - denominator * other,
            denominator
        )
    /**
     * Returns product of the constant represented as rational function and the rational function.
     */
    public override operator fun R.times(other: C): R =
        constructRationalFunction(
            numerator * other,
            denominator
        )

    public override operator fun R.div(other: C): R =
        constructRationalFunction(
            numerator,
            denominator * other
        )

    /**
     * Converts the constant [value] to rational function.
     */
    public override fun number(value: C): R = constructRationalFunction(polynomialNumber(value))

    /**
     * Returns sum of the polynomial represented as rational function and the rational function.
     */
    public override operator fun P.plus(other: R): R =
        constructRationalFunction(
            other.denominator * this + other.numerator,
            other.denominator
        )
    /**
     * Returns difference between the polynomial represented as polynomial and the rational function.
     */
    public override operator fun P.minus(other: R): R =
        constructRationalFunction(
            other.denominator * this - other.numerator,
            other.denominator
        )
    /**
     * Returns product of the polynomial represented as polynomial and the rational function.
     */
    public override operator fun P.times(other: R): R =
        constructRationalFunction(
            this * other.numerator,
            other.denominator
        )

    public override operator fun P.div(other: R): R =
        constructRationalFunction(
            this * other.denominator,
            other.numerator
        )

    /**
     * Returns sum of the polynomial represented as rational function and the rational function.
     */
    public override operator fun R.plus(other: P): R =
        constructRationalFunction(
            numerator + denominator * other,
            denominator
        )
    /**
     * Returns difference between the polynomial represented as rational function and the rational function.
     */
    public override operator fun R.minus(other: P): R =
        constructRationalFunction(
            numerator - denominator * other,
            denominator
        )
    /**
     * Returns product of the polynomial represented as rational function and the rational function.
     */
    public override operator fun R.times(other: P): R =
        constructRationalFunction(
            numerator * other,
            denominator
        )

    public override operator fun R.div(other: P): R =
        constructRationalFunction(
            numerator,
            denominator * other
        )

    /**
     * Converts the polynomial [value] to rational function.
     */
    public override fun number(value: P): R = constructRationalFunction(value)

    /**
     * Returns negation of the rational function.
     */
    public override operator fun R.unaryMinus(): R = constructRationalFunction(-numerator, denominator)
    /**
     * Returns sum of the rational functions.
     */
    public override operator fun R.plus(other: R): R =
        constructRationalFunction(
            numerator * other.denominator + denominator * other.numerator,
            denominator * other.denominator
        )
    /**
     * Returns difference of the rational functions.
     */
    public override operator fun R.minus(other: R): R =
        constructRationalFunction(
            numerator * other.denominator - denominator * other.numerator,
            denominator * other.denominator
        )
    /**
     * Returns product of the rational functions.
     */
    public override operator fun R.times(other: R): R =
        constructRationalFunction(
            numerator * other.numerator,
            denominator * other.denominator
        )

    public override operator fun R.div(other: R): R =
        constructRationalFunction(
            numerator * other.denominator,
            denominator * other.numerator
        )

    /**
     * Instance of zero rational function (zero of the rational functions ring).
     */
    public override val zero: R get() = constructRationalFunction(polynomialZero)

    /**
     * Instance of unit polynomial (unit of the rational functions ring).
     */
    public override val one: R get() = constructRationalFunction(polynomialOne)
}

@Suppress("INAPPLICABLE_JVM_NAME") // FIXME: Waiting for KT-31420
public interface MultivariateRationalFunctionalSpace<
        C,
        V,
        P: Polynomial<C>,
        R: RationalFunction<C, P>
        >: RationalFunctionalSpace<C, P, R> {
    @JvmName("plusVariableInt")
    public operator fun V.plus(other: Int): P
    @JvmName("minusVariableInt")
    public operator fun V.minus(other: Int): P
    @JvmName("timesVariableInt")
    public operator fun V.times(other: Int): P

    @JvmName("plusIntVariable")
    public operator fun Int.plus(other: V): P
    @JvmName("minusIntVariable")
    public operator fun Int.minus(other: V): P
    @JvmName("timesIntVariable")
    public operator fun Int.times(other: V): P

    @JvmName("plusConstantVariable")
    public operator fun C.plus(other: V): P
    @JvmName("minusConstantVariable")
    public operator fun C.minus(other: V): P
    @JvmName("timesConstantVariable")
    public operator fun C.times(other: V): P

    @JvmName("plusVariableConstant")
    public operator fun V.plus(other: C): P
    @JvmName("minusVariableConstant")
    public operator fun V.minus(other: C): P
    @JvmName("timesVariableConstant")
    public operator fun V.times(other: C): P

    @JvmName("unaryPlusVariable")
    public operator fun V.unaryPlus(): P
    @JvmName("unaryMinusVariable")
    public operator fun V.unaryMinus(): P
    @JvmName("plusVariableVariable")
    public operator fun V.plus(other: V): P
    @JvmName("minusVariableVariable")
    public operator fun V.minus(other: V): P
    @JvmName("timesVariableVariable")
    public operator fun V.times(other: V): P

    @JvmName("plusVariablePolynomial")
    public operator fun V.plus(other: P): P
    @JvmName("minusVariablePolynomial")
    public operator fun V.minus(other: P): P
    @JvmName("timesVariablePolynomial")
    public operator fun V.times(other: P): P

    @JvmName("plusPolynomialVariable")
    public operator fun P.plus(other: V): P
    @JvmName("minusPolynomialVariable")
    public operator fun P.minus(other: V): P
    @JvmName("timesPolynomialVariable")
    public operator fun P.times(other: V): P

    @JvmName("plusVariableRational")
    public operator fun V.plus(other: R): R
    @JvmName("minusVariableRational")
    public operator fun V.minus(other: R): R
    @JvmName("timesVariableRational")
    public operator fun V.times(other: R): R

    @JvmName("plusRationalVariable")
    public operator fun R.plus(other: V): R
    @JvmName("minusRationalVariable")
    public operator fun R.minus(other: V): R
    @JvmName("timesRationalVariable")
    public operator fun R.times(other: V): R

    /**
     * Map that associates variables (that appear in the polynomial in positive exponents) with their most exponents
     * in which they are appeared in the polynomial.
     *
     * As consequence all values in the map are positive integers. Also, if the polynomial is constant, the map is empty.
     * And keys of the map is the same as in [variables].
     */
    public val P.degrees: Map<V, UInt>
    /**
     * Counts degree of the polynomial by the specified [variable].
     */
    public fun P.degreeBy(variable: V): UInt = degrees.getOrElse(variable) { 0u }
    /**
     * Counts degree of the polynomial by the specified [variables].
     */
    public fun P.degreeBy(variables: Collection<V>): UInt
    /**
     * Set of all variables that appear in the polynomial in positive exponents.
     */
    public val P.variables: Set<V> get() = degrees.keys
    /**
     * Count of all variables that appear in the polynomial in positive exponents.
     */
    public val P.countOfVariables: Int get() = variables.size

    /**
     * Set of all variables that appear in the polynomial in positive exponents.
     */
    public val R.variables: Set<V> get() = numerator.variables union denominator.variables
    /**
     * Count of all variables that appear in the polynomial in positive exponents.
     */
    public val R.countOfVariables: Int get() = variables.size
}

public interface MultivariateRationalFunctionalSpaceOverRing<
        C,
        V,
        P: Polynomial<C>,
        R: RationalFunction<C, P>,
        A: Ring<C>
        > : RationalFunctionalSpaceOverRing<C, P, R, A>, MultivariateRationalFunctionalSpace<C, V, P, R>

public interface MultivariateRationalFunctionalSpaceOverPolynomialSpace<
        C,
        V,
        P: Polynomial<C>,
        R: RationalFunction<C, P>,
        AP: PolynomialSpace<C, P>,
        > : RationalFunctionalSpaceOverPolynomialSpace<C, P, R, AP>, MultivariateRationalFunctionalSpace<C, V, P, R>

@Suppress("INAPPLICABLE_JVM_NAME") // FIXME: Waiting for KT-31420
public interface MultivariateRationalFunctionalSpaceOverMultivariatePolynomialSpace<
        C,
        V,
        P: Polynomial<C>,
        R: RationalFunction<C, P>,
        AP: MultivariatePolynomialSpace<C, V, P>,
        > : MultivariateRationalFunctionalSpaceOverPolynomialSpace<C, V, P, R, AP> {
    @JvmName("plusVariableInt")
    public override operator fun V.plus(other: Int): P = polynomialRing { this@plus + other }
    @JvmName("minusVariableInt")
    public override operator fun V.minus(other: Int): P = polynomialRing { this@minus - other }
    @JvmName("timesVariableInt")
    public override operator fun V.times(other: Int): P = polynomialRing { this@times * other }

    @JvmName("plusIntVariable")
    public override operator fun Int.plus(other: V): P = polynomialRing { this@plus + other }
    @JvmName("minusIntVariable")
    public override operator fun Int.minus(other: V): P = polynomialRing { this@minus - other }
    @JvmName("timesIntVariable")
    public override operator fun Int.times(other: V): P = polynomialRing { this@times * other }

    @JvmName("plusConstantVariable")
    public override operator fun C.plus(other: V): P = polynomialRing { this@plus + other }
    @JvmName("minusConstantVariable")
    public override operator fun C.minus(other: V): P = polynomialRing { this@minus - other }
    @JvmName("timesConstantVariable")
    public override operator fun C.times(other: V): P = polynomialRing { this@times * other }

    @JvmName("plusVariableConstant")
    public override operator fun V.plus(other: C): P = polynomialRing { this@plus + other }
    @JvmName("minusVariableConstant")
    public override operator fun V.minus(other: C): P = polynomialRing { this@minus - other }
    @JvmName("timesVariableConstant")
    public override operator fun V.times(other: C): P = polynomialRing { this@times * other }

    @JvmName("unaryPlusVariable")
    public override operator fun V.unaryPlus(): P = polynomialRing { +this@unaryPlus }
    @JvmName("unaryMinusVariable")
    public override operator fun V.unaryMinus(): P = polynomialRing { -this@unaryMinus }
    @JvmName("plusVariableVariable")
    public override operator fun V.plus(other: V): P = polynomialRing { this@plus + other }
    @JvmName("minusVariableVariable")
    public override operator fun V.minus(other: V): P = polynomialRing { this@minus - other }
    @JvmName("timesVariableVariable")
    public override operator fun V.times(other: V): P = polynomialRing { this@times * other }

    @JvmName("plusVariablePolynomial")
    public override operator fun V.plus(other: P): P = polynomialRing { this@plus + other }
    @JvmName("minusVariablePolynomial")
    public override operator fun V.minus(other: P): P = polynomialRing { this@minus - other }
    @JvmName("timesVariablePolynomial")
    public override operator fun V.times(other: P): P = polynomialRing { this@times * other }

    @JvmName("plusPolynomialVariable")
    public override operator fun P.plus(other: V): P = polynomialRing { this@plus + other }
    @JvmName("minusPolynomialVariable")
    public override operator fun P.minus(other: V): P = polynomialRing { this@minus - other }
    @JvmName("timesPolynomialVariable")
    public override operator fun P.times(other: V): P = polynomialRing { this@times * other }

    /**
     * Map that associates variables (that appear in the polynomial in positive exponents) with their most exponents
     * in which they are appeared in the polynomial.
     *
     * As consequence all values in the map are positive integers. Also, if the polynomial is constant, the map is empty.
     * And keys of the map is the same as in [variables].
     */
    public override val P.degrees: Map<V, UInt> get() = polynomialRing { degrees }
    /**
     * Counts degree of the polynomial by the specified [variable].
     */
    public override fun P.degreeBy(variable: V): UInt = polynomialRing { degreeBy(variable) }
    /**
     * Counts degree of the polynomial by the specified [variables].
     */
    public override fun P.degreeBy(variables: Collection<V>): UInt = polynomialRing { degreeBy(variables) }
    /**
     * Set of all variables that appear in the polynomial in positive exponents.
     */
    public override val P.variables: Set<V> get() = polynomialRing { variables }
    /**
     * Count of all variables that appear in the polynomial in positive exponents.
     */
    public override val P.countOfVariables: Int get() = polynomialRing { countOfVariables }
}

@Suppress("INAPPLICABLE_JVM_NAME") // FIXME: Waiting for KT-31420
public abstract class MultivariatePolynomialSpaceOfFractions<
        C,
        V,
        P: Polynomial<C>,
        R: RationalFunction<C, P>,
        > : MultivariateRationalFunctionalSpace<C, V, P, R>,  PolynomialSpaceOfFractions<C, P, R>() {
    @JvmName("plusVariableRational")
    public override operator fun V.plus(other: R): R =
        constructRationalFunction(
            this * other.denominator + other.numerator,
            other.denominator
        )
    @JvmName("minusVariableRational")
    public override operator fun V.minus(other: R): R =
        constructRationalFunction(
            this * other.denominator - other.numerator,
            other.denominator
        )
    @JvmName("timesVariableRational")
    public override operator fun V.times(other: R): R =
        constructRationalFunction(
            this * other.numerator,
            other.denominator
        )

    @JvmName("plusRationalVariable")
    public override operator fun R.plus(other: V): R =
        constructRationalFunction(
            numerator + denominator * other,
            denominator
        )
    @JvmName("minusRationalVariable")
    public override operator fun R.minus(other: V): R =
        constructRationalFunction(
            numerator - denominator * other,
            denominator
        )
    @JvmName("timesRationalVariable")
    public override operator fun R.times(other: V): R =
        constructRationalFunction(
            numerator * other,
            denominator
        )
}