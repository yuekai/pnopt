# PNOPT: Proximal Newton OPTimizer

PNOPT (pronounced pee-en-opt) is a MATLAB package that uses proximal Newton-type methods to minimize composite functions. For details, please refer to [Lee et al. (2014)](http://arxiv.org/abs/1206.1623).

## Installation

Unpack the archive and add the `yuekai-PNOPT-xxxxxxx` directory to your MATLAB path, e.g.

    addpath /home/yuekai/matlab/yuekai-PNOPT-xxxxxxx/

We suggest users also install [TFOCS](http://cvxr.com/tfocs/) (pronounced tee-fox), a MATLAB package that uses first-order methods to minimize composite functions (among other things).

## Usage

PNOPT has the calling sequence:

    [ x, f, output ] = pnopt( smoothF, nonsmoothF, x0, options );

The required input arguments are:
* `smoothF`: a smooth function,
* `nonsmoothF`: a nonsmooth function,
* `x0`: a starting point for the solver.

The user can also supply an `options` structure created using `pnopt_optimset` to customize the behavior of PNOPT. `pnopt_optimset` shares a similar interface with MATLAB's `optimset` function:

  options = pnopt_optimset( 'param1', val1, 'param2', val2, ... );

Calling `pnopt_optimset` with no inputs and outputs prints available options.

PNOPT returns:
* `x`: an optimal solution,
* `f`: the optimal value,
* `output`: a structure containing information collected during the execution of PNOPT.

### Creating smooth and nonsmooth functions

Smooth and nonsmooth functions must satisfy these conventions:

* `smoothF(x)` should return the function value and gradient at `x`, i.e. `[ fx, gradx ] = smoothF(x)`,
* `nonsmoothF(x)` should return function value at `x`, i.e. `f_x = nonsmoothF(x)`,
* `nonsmoothF(x,t)` should return the proximal point `y` and the function value at `y`, i.e. `[ f_y, y ] = nonsmoothF(x,t )`.

PNOPT is compatible with the function generators included with TFOCS that accept vector arguments so users can use these generators to create commonly used smooth and nonsmooth functions. Please refer to section 3 of the [TFOCS user guide](https://github.com/cvxr/TFOCS/raw/master/userguide.pdf) for details.

## Demo: sparse logistic regression

The demo requires `LogisticLoss` from [PMTK](https://github.com/probml/pmtk3) and `prox_l1` from TFOCS.

    n = 100;
    p = 200;
    X = randn(n,p);
    y = sign( X * ( (rand(p,1) > .5) .* randn(p,1) ) + randn(n,1) );

    logistic_obj = @(w) LogisticLoss(w,X,y);
    lambda = 10;
    l1_pen  = prox_l1(lambda);
    w0 = zeros(p,1);
    pnopt_options = pnopt_optimset( 'optim', 1e-8 );

    [ w, f ] = pnopt( logistic_obj, l1_pen, w0, pnopt_options );

    ================================================================
                     PNOPT v. 0.9.1 (Dec. 15, 2013)
    ================================================================
             Fun.    Prox     Step len.     Obj. val.        Optim.
    ----------------------------------------------------------------
       0 |      1       0                  6.9315e+01    4.4998e+00
      10 |     11      49    1.0000e+00    6.7911e+01    1.8744e-03
      13 |     14      66    1.0000e+00    6.7911e+01    2.0037e-04
    ----------------------------------------------------------------
