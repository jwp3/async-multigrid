function x = GS_lower(A, x, b, iters)
    M = tril(A);
    for i=1:iters
      r = b - A*x;
      x = x + M\r;
    end
end
