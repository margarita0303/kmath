package space.kscience.kmath.ast.rendering

import space.kscience.kmath.ast.MST

/**
 * Renders [MST] to [MathSyntax].
 *
 * @author Iaroslav Postovalov
 */
public fun interface MathRenderer {
    /**
     * Renders [MST] to [MathSyntax].
     */
    public fun render(mst: MST): MathSyntax
}

/**
 * Implements [MST] render process with sequence of features.
 *
 * @property features The applied features.
 * @author Iaroslav Postovalov
 */
public open class FeaturedMathRenderer(public val features: List<RenderFeature>) : MathRenderer {
    public override fun render(mst: MST): MathSyntax {
        for (feature in features) feature.render(this, mst)?.let { return it }
        throw UnsupportedOperationException("Renderer $this has no appropriate feature to render node $mst.")
    }

    /**
     * Logical unit of [MST] rendering.
     */
    public fun interface RenderFeature {
        /**
         * Renders [MST] to [MathSyntax] in the context of owning renderer.
         */
        public fun render(renderer: FeaturedMathRenderer, node: MST): MathSyntax?
    }
}

/**
 * Extends [FeaturedMathRenderer] by adding post-processing stages.
 *
 * @property stages The applied stages.
 * @author Iaroslav Postovalov
 */
public open class FeaturedMathRendererWithPostProcess(
    features: List<RenderFeature>,
    public val stages: List<PostProcessStage>,
) : FeaturedMathRenderer(features) {
    public override fun render(mst: MST): MathSyntax {
        val res = super.render(mst)
        for (stage in stages) stage.perform(res)
        return res
    }

    /**
     * Logical unit of [MathSyntax] post-processing.
     */
    public fun interface PostProcessStage {
        /**
         * Performs the specified action over [MathSyntax].
         */
        public fun perform(node: MathSyntax)
    }

    public companion object {
        /**
         * The default setup of [FeaturedMathRendererWithPostProcess].
         */
        public val Default: FeaturedMathRendererWithPostProcess = FeaturedMathRendererWithPostProcess(
            listOf(
                // Printing known operations
                BinaryPlus.Default,
                BinaryMinus.Default,
                UnaryPlus.Default,
                UnaryMinus.Default,
                Multiplication.Default,
                Fraction.Default,
                Power.Default,
                SquareRoot.Default,
                Exponential.Default,
                InverseTrigonometricOperations.Default,

                // Fallback option for unknown operations - printing them as operator
                BinaryOperator.Default,
                UnaryOperator.Default,

                // Pretty printing for some objects
                PrettyPrintFloats.Default,
                PrettyPrintIntegers.Default,
                PrettyPrintPi.Default,

                // Printing terminal nodes as string
                PrintNumeric,
                PrintSymbolic,
            ),
            listOf(
                SimplifyParentheses.Default,
                BetterMultiplication,
            ),
        )
    }
}
