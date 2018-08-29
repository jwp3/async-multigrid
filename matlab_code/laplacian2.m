function a = laplacian2(m) 
% a = laplacian2(m) 
%  returns 2D 5-point laplacian matrix of order m^2

% 2011-10-20  EC  initial version, modified from lmatgen

e = -ones(m*m,1);
b1 = e;
b2 = e;
for i = m:m:m*m
   b1(i) = 0;
   b2(i-m+1) = 0;
end
a = spdiags([e b1 -4*e b2 e], [-m,-1,0,1,m], m*m, m*m);
