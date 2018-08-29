function [x, num_relax] = iBlockJacobi(A,x,b,iters,p)
    n = length(A);
    m = n/p;
    
    for i = 1:p
        i_low = ((i-1)*m+1);
        i_high = i*m;
        D(i_low:i_high,i_low:i_high) = tril(A(i_low:i_high,i_low:i_high));
    end
    
    num_relax = 0;
    r = b - A*x;
    for k = 1:iters
        x = x + D\r;        
        r = b - A*x;
        num_relax = num_relax + n;
    end
end