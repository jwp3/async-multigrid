function A = gen2d(m)
% 2016

% m by m grid
A = laplacian2(m);
h = 1/(m+1);
A = A/(h*h);
