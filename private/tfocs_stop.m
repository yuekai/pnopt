function stop = tfocs_stop( x, nonsmoothF, optTol )
  
  global subprob_grad_f_y subprob_optim
  
  [ ~, x_prox ]   = nonsmoothF( x - subprob_grad_f_y, 1 );
    subprob_optim = norm( x_prox - x, 'inf' );
    stop          = subprob_optim <= optTol;
  
    