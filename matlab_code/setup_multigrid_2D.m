function [A, P, R, N, q] = setup_multigrid_2D(a)
    n = length(a);
    q = 1;
    m = n;
    N = n;
    while (m > 9)
        m = (sqrt(m)-1)/2;
        m = m^2;
        N = [N m];
        q = q+1;
    end
    
    R = cell(q,1);
    P = cell(q,1);
    for i = (q-1):-1:1
        P{i} = prolong_2D(sqrt(N(i)));
        R{i+1} = P{i}'/4;
    end
    R{1} = speye(N(1));
    P{end} = speye(N(end));
    
    A = cell(q,1);
    A{1} = a;
    for i = 2:q
        A{i} = R{i}*A{i-1}*P{i-1};
        %A{i} = gen2d(sqrt(N(i)));
    end
end
