function [ x, f_x, output ] = pnopt_pn( smoothF, nonsmoothF, x, options )
% pnopt_pn : Proximal Newton method
% 
% [ x, f, output ] = pnopt_pn( smoothF, nonsmoothF, x, options ) starts at x and 
%   seeks a minimizer of the objective function in composite form. smoothF is a 
%   handle to a function that returns the smooth function value and gradient. 
%   nonsmoothF is a handle to a function that returns the nonsmooth function 
%   value and proximal mapping. options is a structure created using the 
%   pnopt_optimset function.
% 
  REVISION = '$Revision: 0.9.1$';
  DATE     = '$Date: Dec. 15, 2013$';
  REVISION = REVISION(11:end-1);
  DATE     = DATE(8:end-1);
  
% ============ Process options ============
  
  sparsa_options = pnopt_optimset(...
    'display'  , 0    ,...
    'maxfun_evals' , 5000 ,...
    'maxIter'  , 500   ...
    );
  
  tfocs_opts = struct(...
    'alg'        , 'N83' ,...
    'maxIts'     , 500   ,...
    'printEvery' , 0     ,...
    'restart'    , -Inf   ...
    );
  
  debug          = options.debug;
  desc_param     = options.desc_param;
  display        = options.display;
  maxfun_evals   = options.maxfun_evals;
  max_iter        = options.max_iter;
  subprob_solver = options.subprob_solver;
  ftol           = options.ftol;
  optim_tol      = options.optim_tol;
  xtol           = options.xtol;
  
% ------------ Set subproblem solver options ------------
  
  switch subprob_solver
    case 'sparsa'
      sparsa_options = options.sparsa_options;
    case 'tfocs'
      tfocs_opts = options.tfocs_opts;  
      
      if debug
        tfocs_opts.countOps = 1;
        tfocs_opts.errFcn   = @ ( f, x ) tfocs_err();
      end
  end
  
% ============ Initialize variables ============
  
  pnopt_flags
  
  iter         = 0; 
  loop         = 1;
  forcing_term = 0.5;
  
  trace.f_x        = zeros( max_iter + 1, 1 );
  trace.fun_evals  = zeros( max_iter + 1, 1 );
  trace.prox_evals = zeros( max_iter + 1, 1 );
  trace.optim      = zeros( max_iter + 1, 1 );
  
  if debug
    trace.forcing_term    = zeros( max_iter, 1 );
    trace.subprob_iters   = zeros( max_iter, 1 );
    trace.subprob_optim   = zeros( max_iter, 1 );
  end
  
  if display > 0  
    fprintf( ' %s\n', repmat( '=', 1, 64 ) );
    fprintf( '                  PNOPT v.%s (%s)\n', REVISION, DATE );
    fprintf( ' %s\n', repmat( '=', 1, 64 ) );
    fprintf( ' %4s   %6s  %6s  %12s  %12s  %12s \n',...
      '','Fun.', 'Prox', 'Step len.', 'Obj. val.', 'Optim.' );
    fprintf( ' %s\n', repmat( '-', 1, 64 ) );
  end
  
% ------------ Evaluate objective function at starting x ------------
  
  [ g_x, grad_g_x, hess_g_x ] = smoothF( x );
    h_x  = nonsmoothF( x );
    f_x  = g_x + h_x;
  
% ------------ Start collecting data for display and output ------------
  
    fun_evals       = 1;
    prox_evals      = 0;
  [ ~, x_prox ] = nonsmoothF( x - grad_g_x, 1 );
    optim       = norm( x_prox - x, 'inf' );
  
  trace.f_x(1)        = f_x;
  trace.fun_evals(1)  = fun_evals;
  trace.prox_evals(1) = prox_evals;
  trace.optim(1)      = optim; 
  
  if display > 0
    fprintf( ' %4d | %6d  %6d  %12s  %12.4e  %12.4e\n', ...
      iter, fun_evals, prox_evals, '', f_x, optim );
  end
  
% ------------ Check if starting x is optimal ------------
  
  if optim <= optim_tol
    flag    = FLAG_OPTIM;
    message = MESSAGE_OPTIM;
    loop    = 0;
  end

% ============ Main Loop ============
  
  while loop
    iter = iter + 1; 
    
  % ------------ Solve subproblem for a search direction ------------
    
    quadF = @(z) pnopt_quad( hess_g_x, grad_g_x, f_x, z - x );

    switch subprob_solver

      % SpaRSA
      case 'sparsa'
        sparsa_options = pnopt_optimset( sparsa_options,...
          'optim_tol', max( 0.1 * optim_tol, forcing_term * optim ) ...
          );  

        [ x_prox, ~, sparsa_out ] = ...
          pnopt_sparsa( quadF, nonsmoothF, x, sparsa_options ); 

      % ------------ Collect data from subproblem solve ------------

        subprob_iters  = sparsa_out.iters;
        subprob_prox_evals = sparsa_out.prox_evals;

        if debug
          subprob_flag  = sparsa_out.flag; %#ok<NASGU>
          subprob_optim = sparsa_out.optim;
        end

      % TFOCS 
      case 'tfocs'
        tfocs_opts.stopFcn = @(f, x) tfocs_stop( x, nonsmoothF,...
          max( 0.1 * optim_tol, forcing_term * optim ) );

        [ x_prox, tfocs_out ] = ...
          tfocs( quadF, [], nonsmoothF, x, tfocs_opts );

        subprob_iters = tfocs_out.niter;
        if isfield( tfocs_opts, 'countOps' ) && tfocs_opts.countOps
          subprob_prox_evals = tfocs_out.counts(end,5);
        else
          subprob_prox_evals = tfocs_out.niter;
        end

        if debug
          subprob_optim = tfocs_out.err(end);
        end
    end
    
    p = x_prox - x;
    
  % ------------ Conduct line search ------------
    
    x_old      = x;
    f_old      = f_x;
    grad_g_old = grad_g_x;
    optim_old  = optim;
    
    [ x, f_x, grad_g_x, hess_g_x, step, backtrack_flag, backtrack_iters ] = ...
      pnopt_backtrack( x, p, 1, f_x, h_x, grad_g_x' * p, smoothF, nonsmoothF, ...
        desc_param, xtol, maxfun_evals - fun_evals );  %#ok<ASGLU>
    
  % ------------ Select safeguarded forcing term ------------
    
      forcing_term  = min( 0.1, norm( optim - subprob_optim ) / norm( optim_old ) );
    
  % ------------ Collect data for display and output ------------
    
      fun_evals       =  fun_evals + backtrack_iters ;
      prox_evals      = prox_evals + backtrack_iters + subprob_prox_evals;
    [ ~, x_prox ] = nonsmoothF( x - grad_g_x, 1 );
      optim         = norm( x_prox - x, 'inf' );
    
    trace.f_x(iter+1)    = f_x;
    trace.fun_evals(iter+1)  = fun_evals;
    trace.prox_evals(iter+1) = prox_evals;
    trace.optim(iter+1)  = optim;
    
    if debug
      trace.forcing_term(iter)    = forcing_term;
      trace.backtrack_iters(iter) = backtrack_iters;
      trace.subprob_iters(iter)   = subprob_iters;
      trace.subprob_optim(iter)   = subprob_optim;
    end
    
    if display > 0 && mod( iter, display ) == 0
      fprintf( ' %4d | %6d  %6d  %12.4e  %12.4e  %12.4e\n', ...  
        iter, fun_evals, prox_evals, step, f_x, optim );
    end
    
    pnopt_stop
    
  end
  
% ============ Clean up and exit ============
  
  trace.f_x        = trace.f_x(1:iter+1);
  trace.fun_evals  = trace.fun_evals(1:iter+1);
  trace.prox_evals = trace.prox_evals(1:iter+1);
  trace.optim      = trace.optim(1:iter+1);
  
  if debug
    trace.forcing_term    = trace.forcing_term(1:iter);
    trace.backtrack_iters = trace.backtrack_iters(1:iter);
    trace.subprob_iters   = trace.subprob_iters(1:iter);
    trace.subprob_optim   = trace.subprob_optim(1:iter);
  end
  
  if display > 0 && mod( iter, display ) > 0
    fprintf( ' %4d | %6d  %6d  %12.4e  %12.4e  %12.4e\n', ...
      iter, fun_evals, prox_evals, step, f_x, optim );
    fprintf( ' %s\n', repmat( '-', 1, 64 ) );

  end
  
  output = struct( ...    
    'flag'       , flag       ,...
    'fun_evals'  , fun_evals  ,...
    'iters'      , iter       ,...
    'optim'      , optim      ,...
    'options'    , options    ,...
    'prox_evals' , prox_evals ,...
    'trace'      , trace       ...
    );
  
  clear global subprob_Dg_y subprob_optim
  