function [ x, f, output ] = pnopt( smoothF, nonsmoothF, x, options )
% pnopt : Proximal Newton-type methods
% 
% [ x, f, output ] = pnopt( smoothF, nonsmoothF, x ) starts at x and seeks a 
%   minimizer of the objective function in composite form. smoothF is a handle
%   to a function that returns the smooth function value and gradient. nonsmoothF
%   is a handle to a function that returns the nonsmooth function value and 
%   proximal mapping. 
%  
% [ x, f, output ] = pnopt( smoothF, nonsmoothF, x, options ) replaces the default
%   optimization parameters with those in options, a structure created using the
%   pnopt_optimset function.
% 
%   $Revision: 0.8.0 $  $Date: 2012/12/01 $
 
% ============ Process options ============
  
  sparsa_options = pnopt_optimset(...
    'display'       , 0    ,...
    'max_fun_evals' , 5000 ,...
    'max_iter'      , 500   ...
    );
  
  tfocs_opts = struct(...
    'alg'        , 'N83' ,...
    'errFcn'     , @(f,x) tfocs_err() ,...
    'maxIts'     , 500   ,...
    'printEvery' , 0     ,...
    'restart'    , -Inf   ...
    );
  
  if exist( 'tfocs', 'file')
    default_options = pnopt_optimset(...
      'debug'          , 0          ,... % debug mode 
      'desc_param'     , 0.0001     ,... % sufficient descent parameter
      'display'        , 10         ,... % display frequency (<= 0 for no display) 
      'Lbfgs_mem'      , 50         ,... % L-BFGS memory
      'max_fun_evals'  , 5000       ,... % max number of function evaluations
      'max_iter'       , 500        ,... % max number of iterations
      'method'         , 'Lbfgs'    ,... % method for building Hessian approximation
      'subprob_solver' , 'tfocs'    ,... % solver for solving subproblems
      'tfocs_opts'     , tfocs_opts ,... % subproblem solver options
      'ftol'           , 1e-9       ,... % stopping tolerance on relative change in the objective function 
      'optim_tol'      , 1e-6       ,... % stopping tolerance on optimality condition
      'xtol'           , 1e-9        ... % stopping tolerance on solution
      );
    
  else
    default_options = pnopt_optimset(    ...
      'debug'          , 0              ,... 
      'desc_param'     , 0.0001         ,... 
      'display'        , 10             ,... 
      'Lbfgs_mem'      , 50             ,... 
      'max_fun_evals'  , 5000           ,... 
      'max_iter'       , 500            ,... 
      'method'         , 'Lbfgs'        ,... 
      'subprob_solver' , 'sparsa'       ,... 
      'sparsa_options' , sparsa_options ,...
      'ftol'           , 1e-9           ,... 
      'optim_tol'      , 1e-6           ,...
      'xtol'           , 1e-9            ... 
      );
    
  end
  
  if nargin > 3
    if isfield( options, 'sparsa_options' )
      options.sparsa_options = pnopt_optimset( sparsa_options, options.sparsa_options );
    elseif isfield( options, 'tfocs_opts' )
      options.tfocs_opts = merge_struct( tfocs_opts, options.tfocs_opts );
    end
    options = pnopt_optimset( default_options, options );
  else
    options = default_options;
  end
  
  method = options.method;
    
  % ============ Call solver ============
  
  switch method
    case { 'bfgs', 'Lbfgs' }
      [ x, f, output ] = pnopt_pqn( smoothF, nonsmoothF, x, options );
    case 'newton'
      [ x, f, output ] = pnopt_pn( smoothF, nonsmoothF, x, options );
    otherwise
      error( 'Unrecognized method ''%s''.', method ) 
  end
  
  
function S3 = merge_struct( S1 ,S2 )
% merge_struct : merge two structures
%   self-explanatory ^
% 
  S3 = S1;
  S3_names = fieldnames( S2 );
  for k = 1:length( S3_names )
    if isfield( S3, S3_names{k} )
      if isstruct( S3.(S3_names{k}) )
        S3.(S3_names{k}) = merge_struct( S3.(S3_names{k}),...
          S2.(S3_names{k}) );
      else
        S3.(S3_names{k}) = S2.(S3_names{k});
      end
    else
      S3.(S3_names{k}) = S2.(S3_names{k});
    end
end
  
  