function a = laplacian1(n)
    e = -ones(n,1);
    a = spdiags([e -2*e e], [-1,0,1], n, n);
end
