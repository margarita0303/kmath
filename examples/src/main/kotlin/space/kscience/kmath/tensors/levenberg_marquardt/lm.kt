package space.kscience.kmath.tensors.levenberg_marquardt

import space.kscience.kmath.ejml.EjmlLinearSpaceDDRM.solve
import space.kscience.kmath.linear.transpose
import space.kscience.kmath.nd.DoubleFieldOpsND.Companion.minus
import space.kscience.kmath.nd.MutableStructure2D
import space.kscience.kmath.nd.as1D
import space.kscience.kmath.nd.as2D
import space.kscience.kmath.nd4j.DoubleNd4jArrayFieldOps.Companion.times
import space.kscience.kmath.nd4j.DoubleNd4jTensorAlgebra.max
import space.kscience.kmath.real.max
import space.kscience.kmath.tensors.core.BroadcastDoubleTensorAlgebra.div
import space.kscience.kmath.tensors.core.BroadcastDoubleTensorAlgebra.dot
import space.kscience.kmath.tensors.core.BroadcastDoubleTensorAlgebra.fromArray
import space.kscience.kmath.tensors.core.BroadcastDoubleTensorAlgebra.ones
import space.kscience.kmath.tensors.core.BroadcastDoubleTensorAlgebra.variance
import space.kscience.kmath.tensors.core.BroadcastDoubleTensorAlgebra.zeros
import space.kscience.kmath.tensors.core.DoubleTensorAlgebra.Companion.copy
import space.kscience.kmath.tensors.core.DoubleTensorAlgebra.Companion.plus
import space.kscience.kmath.tensors.core.DoubleTensorAlgebra.Companion.transpose
import kotlin.math.pow

var iteration: Int = 0  // global variable
var func_calls: Int = 0 // global variable
var example_number = 3 // global variable

val eps = 2.2204e-16    // matlab constant

/* nargin = number of not empty parameters */
fun lm(func: (MutableStructure2D<Double>,  MutableStructure2D<Double>) ->  MutableStructure2D<Double>,
       p_input: MutableStructure2D<Double>, t_input: MutableStructure2D<Double>, y_dat_input: MutableStructure2D<Double>,
       weight_input: MutableStructure2D<Double>, dp_input: MutableStructure2D<Double>, p_min_input: MutableStructure2D<Double>, p_max_input: MutableStructure2D<Double>,
       c_input: MutableStructure2D<Double>, opts_input: DoubleArray, nargin: Int) {

    val tensor_parameter = 0

    iteration  = 0 // iteration counter
    func_calls = 0 // running count of function evaluations

    var p = p_input
    val y_dat = y_dat_input
    val t = t_input

    val Npar   = length(p)                           // number of parameters
    val Npnt   = length(y_dat)                       // number of data points
    var p_old = zeros(intArrayOf(Npar, 1)).as2D()    // previous set of parameters
    var y_old = zeros(intArrayOf(Npnt, 1)).as2D()    // previous model, y_old = y_hat(t;p_old)
    var X2 = 1e-3 / eps                              // a really big initial Chi-sq value
    var X2_old = 1e-3 / eps                          // a really big initial Chi-sq value
    var J = zeros(intArrayOf(Npnt, Npar)).as2D()     // Jacobian matrix
    val DoF = Npnt - Npar + 1                        // statistical degrees of freedom

    var corr_p = 0
    var sigma_p = 0
    var sigma_y = 0
    var R_sq = 0
    var cvg_hist = 0

    if (length(t) != length(y_dat)) {
        println("lm.m error: the length of t must equal the length of y_dat")
        val length_t = length(t)
        val length_y_dat = length(y_dat)
        X2 = 0.0

        corr_p = 0
        sigma_p = 0
        sigma_y = 0
        R_sq = 0
        cvg_hist = 0

        if (tensor_parameter != 0) { // Зачем эта проверка?
            return
        }
    }

    var weight = weight_input
    if (nargin <  5) {
        weight = fromArray(intArrayOf(1, 1), doubleArrayOf((y_dat.transpose().dot(y_dat)).as1D()[0])).as2D()
    }

    var dp = dp_input
    if (nargin < 6) {
        dp = fromArray(intArrayOf(1, 1), doubleArrayOf(0.001)).as2D()
    }

    var p_min = p_min_input
    if (nargin < 7) {
        p_min = p
        p_min.abs()
        p_min = p_min.div(-100.0).as2D()
    }

    var p_max = p_max_input
    if (nargin < 8) {
        p_max = p
        p_max.abs()
        p_max = p_max.div(100.0).as2D()
    }

    var c = c_input
    if (nargin < 9) {
        c = fromArray(intArrayOf(1, 1), doubleArrayOf(1.0)).as2D()
    }

    var opts = opts_input
    if (nargin < 10) {
        opts = doubleArrayOf(3.0, 10.0 * Npar, 1e-3, 1e-3, 1e-1, 1e-1, 1e-2, 11.0, 9.0, 1.0)
    }

    val prnt          = opts[0]                // >1 intermediate results; >2 plots
    val MaxIter       = opts[1].toInt()        // maximum number of iterations
    val epsilon_1     = opts[2]                // convergence tolerance for gradient
    val epsilon_2     = opts[3]                // convergence tolerance for parameters
    val epsilon_3     = opts[4]                // convergence tolerance for Chi-square
    val epsilon_4     = opts[5]                // determines acceptance of a L-M step
    val lambda_0      = opts[6]                // initial value of damping paramter, lambda
    val lambda_UP_fac = opts[7]                // factor for increasing lambda
    val lambda_DN_fac = opts[8]                // factor for decreasing lambda
    val Update_Type   = opts[9].toInt()        // 1: Levenberg-Marquardt lambda update
    // 2: Quadratic update
    // 3: Nielsen's lambda update equations

    val plotcmd = "figure(102); plot(t(:,1),y_init,''-k'',t(:,1),y_hat,''-b'',t(:,1),y_dat,''o'',''color'',[0,0.6,0],''MarkerSize'',4); title(sprintf(''\\chi^2_\\nu = %f'',X2/DoF)); drawnow"

    p_min = make_column(p_min)
    p_max = make_column(p_max)

    if (length(make_column(dp)) == 1) {
        dp = ones(intArrayOf(Npar, 1)).div(1 / dp[0, 0]).as2D()
    }

    val idx = get_zero_indices(dp)                 // indices of the parameters to be fit
    val Nfit = idx?.shape?.component1()            // number of parameters to fit
    var stop = false                               // termination flag
    val y_init = feval(func, t, p)                 // residual error using p_try

    if (weight.shape.component1() == 1 || weight.variance() == 0.0) { // identical weights vector
        weight = ones(intArrayOf(Npnt, 1)).div(1 / kotlin.math.abs(weight[0, 0])).as2D() // !!! need to check
        println("using uniform weights for error analysis")
    }
    else {
        weight = make_column(weight)
        weight.abs()
    }

    // initialize Jacobian with finite difference calculation
    var lm_matx_ans = lm_matx(func, t, p_old, y_old,1, J, p, y_dat, weight, dp)
    var JtWJ = lm_matx_ans[0]
    var JtWdy = lm_matx_ans[1]
    X2 = lm_matx_ans[2][0, 0]
    var y_hat = lm_matx_ans[3]
    J = lm_matx_ans[4]

    if ( abs(JtWdy).max()!! < epsilon_1 ) {
        println(" *** Your Initial Guess is Extremely Close to Optimal ***\n")
        println(" *** epsilon_1 = %e\n$epsilon_1")
        stop = true
    }

    var lambda = 1.0
    var nu = 1
    when (Update_Type) {
        1 -> lambda  = lambda_0                       // Marquardt: init'l lambda
        else -> {                                     // Quadratic and Nielsen
            lambda  = lambda_0 * (diag(JtWJ)).max()!!
            nu = 2
        }
    }

    X2_old = X2 // previous value of X2
    var cvg_hst = ones(intArrayOf(MaxIter, Npar + 3))         // initialize convergence history

//    cvg_hst.as2D().print()

    var h = JtWJ.copy()
    var dX2 = X2
    while (!stop && iteration <= MaxIter) {                   //--- Start Main Loop
        iteration += 1

        // incremental change in parameters
        h = when (Update_Type) {
            1 -> {                // Marquardt
                val solve = solve(JtWJ.plus(make_matrx_with_diagonal(diag(JtWJ)).div(1 / lambda)).as2D(), JtWdy)
                solve.copy()
            }

            else -> {             // Quadratic and Nielsen
                val solve = solve(JtWJ.plus(lm_eye(Npar).div(1 / lambda)).as2D(), JtWdy)
                solve.copy()
            }
        }

        // big = max(abs(h./p)) > 2;                      % this is a big step

        // --- Are parameters [p+h] much better than [p] ?

        var p_try = (p + h).as2D()  // update the [idx] elements
        p_try = smallest_element_comparison(largest_element_comparison(p_min, p_try.as2D()), p_max)  // apply constraints

        var delta_y = y_dat.minus(feval(func, t, p_try))   // residual error using p_try

        // TODO
        //if ~all(isfinite(delta_y))                     // floating point error; break
        //    stop = 1;
        //    break
        //end

        func_calls += 1

        val tmp = delta_y.times(weight)
        var X2_try = delta_y.as2D().transpose().dot(tmp)     // Chi-squared error criteria

        val alpha = 1.0
        if (Update_Type == 2) { // Quadratic
            // One step of quadratic line update in the h direction for minimum X2

//            TODO
//            val alpha =  JtWdy.transpose().dot(h)  / ((X2_try.minus(X2)).div(2.0).plus(2 * JtWdy.transpose().dot(h)))
//            alpha =  JtWdy'*h / ( (X2_try - X2)/2 + 2*JtWdy'*h ) ;
//            h = alpha * h;
//
//            p_try = p + h(idx);                          % update only [idx] elements
//            p_try = min(max(p_min,p_try),p_max);         % apply constraints
//
//            delta_y = y_dat - feval(func,t,p_try,c);     % residual error using p_try
//            func_calls = func_calls + 1;
//            тX2_try = delta_y' * ( delta_y .* weight );   % Chi-squared error criteria
        }

        val rho = when (Update_Type) { // Nielsen
            1 -> {
                val tmp = h.transpose().dot(make_matrx_with_diagonal(diag(JtWJ)).div(1 / lambda).dot(h).plus(JtWdy))
                X2.minus(X2_try).as2D()[0, 0] / abs(tmp.as2D()).as2D()[0, 0]
            }
            else -> {
                val tmp = h.transpose().dot(h.div(1 / lambda).plus(JtWdy))
                X2.minus(X2_try).as2D()[0, 0] / abs(tmp.as2D()).as2D()[0, 0]
            }
        }

        println()
        println("rho = " + rho)

        if (rho > epsilon_4) { // it IS significantly better
            val dX2 = X2.minus(X2_old)
            X2_old = X2
            p_old = p.copy().as2D()
            y_old = y_hat.copy().as2D()
            p = make_column(p_try) // accept p_try

            lm_matx_ans = lm_matx(func, t, p_old, y_old, dX2.toInt(), J, p, y_dat, weight, dp)
            // decrease lambda ==> Gauss-Newton method

            JtWJ = lm_matx_ans[0]
            JtWdy = lm_matx_ans[1]
            X2 = lm_matx_ans[2][0, 0]
            y_hat = lm_matx_ans[3]
            J = lm_matx_ans[4]

            lambda = when (Update_Type) {
                1 -> { // Levenberg
                    kotlin.math.max(lambda / lambda_DN_fac, 1e-7);
                }
                2 -> { // Quadratic
                    kotlin.math.max( lambda / (1 + alpha) , 1e-7 );
                }
                else -> { // Nielsen
                    nu = 2
                    lambda * kotlin.math.max( 1.0 / 3, 1 - (2 * rho - 1).pow(3) )
                }
            }

            // if (prnt > 2) {
            //    eval(plotcmd)
            // }
        }
        else { // it IS NOT better
            X2 = X2_old // do not accept p_try
            if (iteration % (2 * Npar) == 0 ) { // rank-1 update of Jacobian
                lm_matx_ans = lm_matx(func, t, p_old, y_old,-1, J, p, y_dat, weight, dp)
                JtWJ = lm_matx_ans[0]
                JtWdy = lm_matx_ans[1]
                dX2 = lm_matx_ans[2][0, 0]
                y_hat = lm_matx_ans[3]
                J = lm_matx_ans[4]
            }

            // increase lambda  ==> gradient descent method
            lambda = when (Update_Type) {
                1 -> { // Levenberg
                    kotlin.math.min(lambda * lambda_UP_fac, 1e7)
                }
                2 -> { // Quadratic
                    lambda + kotlin.math.abs(((X2_try.as2D()[0, 0] - X2) / 2) / alpha)
                }
                else -> { // Nielsen
                    nu *= 2
                    lambda * (nu / 2)
                }
            }
//            println("\nLAMBDA + " + lambda + "\n")
        }

        if (prnt > 1) {
            val chi_sq = X2 / DoF
            println("Iteration $iteration, func_calls $func_calls | chi_sq=$chi_sq | lambda=$lambda")
            print("param: ")
            for (pn in 0 until Npar) {
                print(p[pn, 0].toString() + " ")
            }
            print("\ndp/p: ")
            for (pn in 0 until Npar) {
                print((h.as2D()[pn, 0] / p[pn, 0]).toString() + " ")
            }
        }

        // update convergence history ... save _reduced_ Chi-square
        // cvg_hst(iteration,:) = [ func_calls  p'  X2/DoF lambda ];

        if (abs(JtWdy).max()!! < epsilon_1 && iteration > 2) {
            println(" **** Convergence in r.h.s. (\"JtWdy\")  ****")
            println(" **** epsilon_1 = $epsilon_1")
            stop = true
        }
        if ((abs(h.as2D()).div(abs(p) + 1e-12)).max() < epsilon_2  &&  iteration > 2) {
            println(" **** Convergence in Parameters ****")
            println(" **** epsilon_2 = $epsilon_2")
            stop = true
        }
        if (X2 / DoF < epsilon_3 && iteration > 2) {
            println(" **** Convergence in reduced Chi-square  **** ")
            println(" **** epsilon_3 = $epsilon_3")
            stop = true
        }
        if (iteration == MaxIter) {
            println(" !! Maximum Number of Iterations Reached Without Convergence !!")
            stop = true
        }
    }  // --- End of Main Loop

    // --- convergence achieved, find covariance and confidence intervals

    // ---- Error Analysis ----

//    if (weight.shape.component1() == 1 || weight.variance() == 0.0) {
//        weight = DoF / (delta_y.transpose().dot(delta_y)) * ones(intArrayOf(Npt, 1))
//    }

//    if (nargout > 1) {
//        val redX2 = X2 / DoF
//    }
//
//    lm_matx_ans = lm_matx(func, t, p_old, y_old, -1, J, p, y_dat, weight, dp)
//    JtWJ = lm_matx_ans[0]
//    JtWdy = lm_matx_ans[1]
//    X2 = lm_matx_ans[2][0, 0]
//    y_hat = lm_matx_ans[3]
//    J = lm_matx_ans[4]
//
//    if (nargout > 2) { // standard error of parameters
//        covar_p = inv(JtWJ);
//        siif nagma_p = sqrt(diag(covar_p));
//    }
//
//    if (nargout > 3) { // standard error of the fit
//        ///  sigma_y = sqrt(diag(J * covar_p * J'));        // slower version of below
//        sigma_y = zeros(Npnt,1);
//        for i=1:Npnt
//            sigma_y(i) = J(i,:) * covar_p * J(i,:)';
//        end
//        sigma_y = sqrt(sigma_y);
//    }
//
//    if (nargout > 4) { // parameter correlation matrix
//        corr_p = covar_p ./ [sigma_p*sigma_p'];
//    }
//
//    if (nargout > 5) { // coefficient of multiple determination
//        R_sq = corr([y_dat y_hat]);
//        R_sq = R_sq(1,2).^2;
//    }
//
//    if (nargout > 6) { // convergence history
//        cvg_hst = cvg_hst(1:iteration,:);
//    }
}