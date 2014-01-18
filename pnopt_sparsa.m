function [ x, f_x, output ] = pnopt_sparsa( smoothF, nonsmoothF, x, options )
% pnopt_sparsa : Structured reconstruction by separable approximation (SpaRSA)
% 
% [ x, f, output ] = pnopt_sparsa( smoothF, nonsmoothF, x, options ) starts at x 
%   and seeks a minimizer of the objective function in composite form. smoothF  
%   is a handle to a function that returns the smooth function value and  
%   gradient. nonsmoothF is a handle to a function that returns the nonsmooth  
%   function value and proximal mapping. options is a structure created using  
%   the pnopt_optimset function.
% 
  REVISION = '$Revision: 0.8.4$';
  DATE     = '$Date: Jun. 30, 2013';
  REVISION = REVISION(11:end-1);
  DATE     = DATE(8:end-1);
  
% ============ Process options ============
  
  default_options = pnopt_optimset(...
    'debug'         , 0      ,... % debug mode 
    'desc_param'    , 0.0001 ,... % sufficient descent parameter
    'display'       , 100    ,... % display frequency (<= 0 for no display) 
    'backtrack_mem' , 10     ,... % number of previous function values to save
    'max_fun_evals' , 50000  ,... % max number of function evaluations
    'max_iter'      , 5000   ,... % max number of iterations
    'ftol'          , 1e-9   ,... % stopping tolerance on objective function 
    'optim_tol'     , 1e-6   ,... % stopping tolerance on opt
    'xtol'          , 1e-9    ... % stopping tolerance on solution
    );
  
  if nargin > 3
    options = pnopt_optimset( default_options, options );
  else
    options = default_options;
  end
  
  debug         = options.debug;
  desc_param    = options.desc_param;
  display       = options.display;
  backtrack_mem = options.backtrack_mem;
  max_fun_evals = options.max_fun_evals;
  max_iter      = options.max_iter;
  ftol          = options.ftol;
  optim_tol     = options.optim_tol;
  xtol          = options.xtol;
  
% ============ Initialize variables ============
  
  pnopt_flags
  
  iter = 0; 
  loop = 1;
  
  trace.f_x    = zeros( max_iter + 1, 1 );
  trace.fun_evals  = zeros( max_iter + 1, 1 );
  trace.prox_evals = zeros( max_iter + 1, 1 );
  trace.optim  = zeros( max_iter + 1, 1 );
  
  if debug
    trace.normDx          = zeros( max_iter, 1 );
    trace.backtrack_flag  = zeros( max_iter, 1 );
    trace.backtrack_iters = zeros( max_iter, 1 );
  end
  
  if display > 0    
    fprintf( ' %s\n', repmat( '=', 1, 64 ) );
    fprintf( '                 SpaRSA v.%s (%s)\n', REVISION, DATE );
    fprintf( ' %s\n', repmat( '=', 1, 64 ) );
    fprintf( ' %4s   %6s  %6s  %12s  %12s  %12s \n',...
      '','Fun.', 'Prox', 'Step len.', 'Obj. val.', 'Optim.' );
    fprintf( ' %s\n', repmat( '=', 1, 64 ) );
  end
  
% ------------ Evaluate objective function at starting x ------------
  
  [ g_x, grad_g_x ] = smoothF(x);
    h_x         = nonsmoothF(x);
    f_x         = g_x + h_x;
  
% ------------ Start collecting data for display and output ------------
  
    fun_evals   = 1;
    prox_evals  = 0;
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
    
  % ------------ Compute search direction ------------
    
    if iter > 1
      s  = x - x_old;
      y  = grad_g_x - grad_f_old;
      BBstep  = ( y' * s ) / ( y' * y );
      if BBstep <= 1e-9 || 1e9 <= BBstep
        BBstep = min( 1, 1 / norm( grad_g_x, 1 ) );
      end
    else
      BBstep = min( 1, 1 / norm( grad_g_x, 1) );
    end
    
  % ------------ Conduct line search ------------
    
    x_old   = x;
    if iter+1 > backtrack_mem
      f_old = [f_old(2:end), f_x];
    else
      f_old(iter) = f_x;
    end
    grad_f_old  = grad_g_x;
    
    [ x, f_x, grad_g_x, step, curvtrack_flag ,curvtrack_iters ] = ...
      pnopt_curvtrack( x, - grad_g_x, BBstep, f_old, - norm(grad_g_x) ^2, smoothF, ...
        nonsmoothF, desc_param, xtol, max_fun_evals - fun_evals ); 
    
  % ------------ Collect data and display status ------------
    
      fun_evals   = fun_evals + curvtrack_iters;
      prox_evals  = prox_evals + curvtrack_iters;
    [ ~, x_prox ] = nonsmoothF( x - grad_g_x ,1);
      optim       = norm( x_prox - x ,'inf');
    
    trace.f_x(iter+1)        = f_x;
    trace.fun_evals(iter+1)  = fun_evals;
    trace.prox_evals(iter+1) = prox_evals;
    trace.optim(iter+1)      = optim; 
    
    if debug
      trace.backtrack_flag(iter)  = curvtrack_flag;
      trace.backtrack_iters(iter) = curvtrack_iters;
    end
    
    if display > 0 && mod( iter, display ) == 0
      fprintf( ' %4d | %6d  %6d  %12.4e  %12.4e  %12.4e\n', ...
        iter, fun_evals, prox_evals, step, f_x, optim );
    end
    
    pnopt_stop
  
  end
  
% ============ Cleanup and exit ============
  
  trace.f_x        = trace.f_x(1:iter+1);
  trace.fun_evals  = trace.fun_evals(1:iter+1);
  trace.prox_evals = trace.prox_evals(1:iter+1);
  trace.optim      = trace.optim(1:iter+1);
  
  if debug
    trace.backtrack_flag  = trace.backtrack_flag(1:iter);
    trace.backtrack_iters = trace.backtrack_iters(1:iter);
  end
  
  if display > 0 && mod(iter,display) > 0
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
  
