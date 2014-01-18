function varargout = pnopt_quad( P, q, r, x )
  
  global subprob_grad_f_y
  
  grad_f_y = P(x) + q;
  varargout{1} = 0.5 * x' * ( grad_f_y + q ) + r;
  subprob_grad_f_y = grad_f_y;
  
  if nargout > 1
    varargout{2} = grad_f_y;
  end
  