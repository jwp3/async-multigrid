function x = Jacobi(A, x, b, iters, omega)
    M = omega./diag(A);
    for i = 1:iters
      r = b - A*x;
      x = x + M.*r;
    end
end
