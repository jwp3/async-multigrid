function p = prolong(n)
% return 2D prolongation operator for a fine grid of n by n
% using bilinear interpolation

e = ones(n,1);
x = spdiags([e -2*e e], -1:1, n, n);
p = 0.25*abs(kron(x,x)); % n^2 by n^2 matrix

% select columns corresponding to coarse grid points

% corners are coarse grid points % changed
% n should be odd

% coarse grid is m by m
m = (n-1)/2;  % changed
cpts = zeros(m*m,1);
k = 0;
for i=2:2:n
    for j=2:2:n
        k = k + 1;
        cpts(k) = (i-1)*n + j;
    end
end

p = p(:,cpts);


