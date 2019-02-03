function [A, P, R, N, num_levels, f_points, c_points, B, G] = setup_multigrid_2D(a)
    global mg_type;
    global omega;

    n = length(a);
    num_levels = 1;
    m = n;
    N = n;
    while (m > 9)
        m = (sqrt(m)-1)/2;
        m = m^2;
        N = [N m];
        num_levels = num_levels+1;
    end
    
    R = cell(num_levels,1);
    P = cell(num_levels,1);
    for k = (num_levels-1):-1:1
        P{k} = prolong_2D(sqrt(N(k)));
        R{k+1} = P{k}'/4;
%         R{k+1} = restrict_inject_2D(sqrt(N(k)));
%         P{k} = R{k+1}';
    end
    R{1} = speye(N(1));
    P{end} = speye(N(end));
    
    A = cell(num_levels,1);
    A{1} = a;
    for k = 2:num_levels
        A{k} = R{k}*A{k-1}*P{k-1};
        %A{i} = gen2d(snum_levelsrt(N(i)));
    end
    G = cell(num_levels,1);
%     for k = 1:num_levels
%         if (strcmp(mg_type, 'multadd') == 1)
%             I = speye(N(k));
%             G{k} = (I - sparse(diag(1./diag(A{k})))*A{k});
%         end
%    end
    B = cell(num_levels,1);
%     for k = 1:num_levels
%         if (strcmp(mg_type, 'multadd') == 1)
%             B{k} = speye(N(1));
%             for i = 1:(k-1)
%                 B{k} = R{i+1}*G{i}*B{k};
%             end
%             if (k < num_levels)
%                 Di = diag(1./diag(A{k}));
%                 D = diag(diag(A{k}));
%                 B{k} = Di*(2*D - A{k})*Di*B{k};
%                 %B{k} = Di*B{k};
%             else
%                 B{k} = A{k}\B{k};
%             end
%             for i = (k-1):-1:1
%                 B{k} = G{i}*P{i}*B{k};
%             end
%         end
%     end

    f_points = cell(num_levels,1);
    c_points = cell(num_levels,1);
    for k = 1:num_levels
        m = sqrt(N(k));
        f_points{k} = 1:m;
        c_points{k} = [];
        j = m+1;
        for i = 2:(m-1)
            if (mod(i,2) == 0)
                f_points{k} = [f_points{k} j:2:(j+m)];
                c_points{k} = [c_points{k} (j+1):2:(j+m-1)];
            else
                f_points{k} = [f_points{k} j:(j+m)];
            end
            j = j + m;
        end
        f_points{k} = [f_points{k} j:N(k)];
    end
end
