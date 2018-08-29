function A = gen1d(n)
    A = laplacian1(n);
    h = 1/(n+1);
    A = A/h;
end
