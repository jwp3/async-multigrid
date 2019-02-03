function [A, P, R, N, num_levels, f_points, c_points, B, G] = setup_multigrid_1D(a)
    n = length(a);
    num_levels = 1;
    N = n;
    while (n > 3)
        n = (n-1)/2;
        N = [N n];
        num_levels = num_levels+1;
    end
    R = cell(num_levels,1);
    P = cell(num_levels,1);
    for k = (num_levels-1):-1:1 
        P{k} = prolong_1D(N(k));
        R{k+1} = P{k}'/2;
%         R{k+1} = restrict_inject_1D(N(k));
%         P{k} = R{k+1}';
    end
    A = cell(num_levels,1);
    A{1} = a;
    for k = 2:num_levels
        A{k} = R{k}*A{k-1}*P{k-1};
        %A{i} = gen1d(N(i));
    end
    G = cell(num_levels,1);
    for k = 1:num_levels
        I = speye(N(k));
        G{k} = (I - sparse(diag(1./diag(A{k})))*A{k});
    end
    B = cell(num_levels,1);
    for k = 1:num_levels
        B{k} = speye(N(1));
        for i = 1:(k-1)
            B{k} = R{i+1}*G{i}*B{k};
        end
        if (k < num_levels)
            Di = diag(1./diag(A{k}));
            D = diag(diag(A{k}));
            %B{k} = Di*(2*D - A{k})*Di*B{k};
            B{k} = Di*B{k};
        else
            B{k} = A{k}\B{k};
        end
        for i = (k-1):-1:1
            B{k} = G{i}*P{i}*B{k};
        end
    end
    
    f_points = cell(num_levels,1);
    c_points = cell(num_levels,1);
    for k = 1:num_levels
        f_points{k} = 1:2:N(k);
        c_points{k} = 2:2:(N(k)-1);
    end
end
