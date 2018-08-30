function [u, iter, model_time, solve_hist] = async_Jacobi(A, u, b, max_iter, max_relax, max_relax_wait, max_read_delay, omega)
    global async_flag;
    global async_type;
    global smoother_print_flag;
    
    n = length(A);
    [row,col,a] = find(A);
    d = diag(A);
    
    %if (strcmp(async_type, 'general') == 1)
        last_read = ones(n,1);
    %end
    
    if (async_flag == 0)
        relax_wait = max(randi([0 max_relax_wait]))*ones(n,1);
    else
        relax_wait = randi([0 max_relax_wait],n,1);
    end
    
    r = (b - A*u);
    
    model_time = 0;
    r0 = r;
    r0_norm = norm(r0);
    
    relax_time_count = zeros(n,1);
    relax_count = zeros(n,1);
    
    solve_hist = zeros(max_iter+1, 3);
    solve_hist(1,:) = [0 0 1];
    
    iter = 0;
    u_hist = u;
    
    while (iter < max_iter)
        relax_flag = zeros(n,1);
        while(1)
            for i = 1:n
                if ((relax_time_count(i) == relax_wait(i)) && (relax_count(i) < max_relax))
                    
                    coli = col(row == i);
                    ai = a(row == i);
                    
                    randi_high = model_time+1;
                    %randi_low = max([last_read(i) model_time-max_read_delay+1]);
                    %last_read(i) = iter_read;
                    randi_low = max([1 model_time-max_read_delay+1]);
                    iter_read = randi([randi_low randi_high],length(coli),1)';
                    
                    u_read = zeros(length(coli),1);
                    for j = 1:length(coli)
                        u_read(j) = u_hist(coli(j),iter_read(j));
                    end
                    
                    s = sum(u_read .* ai);

                    u(i) = omega*(b(i) - s + d(i)*u(i))/d(i) + (1 - omega)*u(i);
                                
                    relax_flag(i) = 1;
                    relax_count(i) = relax_count(i) + 1;
                end
                if (relax_count(i) == max_relax)
                    relax_flag(i) = 1;
                end
            end
            
            u_hist = [u_hist u];
            model_time = model_time + 1;
            
            for i = 1:n
                if (relax_time_count(i) < relax_wait(i))
                    relax_time_count(i) = relax_time_count(i) + 1;
                else
                    if (async_flag == 1)
                        relax_wait(i) = randi([0 max_relax_wait]);
                    end
                    relax_time_count(i) = 0;
                end
            end
            if (sum(relax_flag) == n)
                break;
            end
        end
        if (async_flag == 0)
            relax_wait = max(randi([0 max_relax_wait],n,1))*ones(n,1);
        end
        r = (b - A*u);
        iter = iter + 1;
        solve_hist(iter+1,1) = iter;
        solve_hist(iter+1,2) = model_time;
        solve_hist(iter+1,3) = norm(r)/r0_norm;
        if (smoother_print_flag == 1)
            fprintf('%2d %2d %e\n', ...
                     iter, model_time, norm(r)/r0_norm);
        end
    end
end