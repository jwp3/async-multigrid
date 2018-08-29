function [A, P, R, N, q] = setup_multigrid_1D(a)
    n = length(a);
    q = 1;
    N = n;
    while (n > 3)
        n = (n-1)/2;
        N = [N n];
        q = q+1;
    end
    R = cell(q,1);
    P = cell(q,1);
    for i = (q-1):-1:1
        P{i} = prolong_1D(N(i));
        R{i+1} = P{i}'/2;
    end
    A = cell(q,1);
    A{1} = a;
    for i = 2:q
        A{i} = R{i}*A{i-1}*P{i-1};
        %A{i} = gen1d(N(i));
    end
end
