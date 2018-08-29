function p = prolong_1D(n)

e = ones(n,1);
p = spdiags([e 2*e e], -1:1, n, n);

% select columns corresponding to coarse grid points

% corners are coarse grid points % changed
% n should be odd

% coarse grid is m by m
m = (n-1)/2;  % changed
cpts = zeros(m,1);
k = 0;
for i=2:2:n
    k = k + 1;
    cpts(k) = i;
end
p = p(:,cpts)/2;


