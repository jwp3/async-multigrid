function [u, model_time, grid_wait_list, solve_hist, num_correct] = ...
    multigrid_add_async_delaygrid(A, u, b, P, R, N, q, max_iter, num_relax, max_grid_wait, max_grid_read_delay, max_smooth_wait, max_smooth_read_delay, grid_wait_list)

    global smooth_type;
    global async_flag;
    global mg_type;
    global async_type;
    global omega;
    global print_flag;

    iter = 0;
    d = cell(q,1);
    G = cell(q,1);
    I = cell(q,1);
    e_write = cell(q,1);
    
    if (strcmp(async_type, 'full-async') == 1)
        last_correct_read = cell(q,1);
    elseif (strcmp(async_type, 'semi-async') == 1)
        last_correct_read = ones(q,1);
    end
    
    u_hist = u;
    
    r = (b - A{1}*u);
    for k = 1:q     
        if (strcmp(async_type, 'full-async') == 1)
            last_correct_read{k} = ones(N(1),1);
        end
        
        if (strcmp(mg_type, 'multadd') == 1)
            d{k} = (1./diag(A{k}));

            I{k} = speye(N(k));
            G{k} = (I{k} - omega*spdiags(d{k},0,N(k),N(k))*A{k});
        end
    end
    
    model_time = 0;
    num_correct = 0;
    r0 = r;
    r0_norm = norm(r0);

    if (async_flag == 0)
        if (grid_wait_list(iter+1) > -1)
            grid_wait = grid_wait_list(1)*ones(q,1);
        else
            grid_wait = max(randi([0 max_grid_wait],q,1))*ones(q,1);
        end
    else
        grid_wait = randi([0 max_grid_wait],q,1);
    end
    
    grid_time_count = zeros(q,1);
    
    solve_hist = zeros(max_iter+1, 3);
    solve_hist(1,:) = [0 0 1];
    
    while (iter < max_iter)
        grid_flag = zeros(q,1);
        prev_model_time = model_time;
        while(1)
            c_ind = [];
            for k = 1:q
                if (grid_time_count(k) == grid_wait(k))
                    grid_flag(k) = 1;
                    c_ind = [c_ind k];
                    if (async_flag == 1)
                        u_read = zeros(N(1),1);
                        if (strcmp(async_type, 'full-async') == 1)
                            %randi_low = num_correct-max_grid_read_delay+1;
                            randi_high = num_correct+1;
                            for i = 1:N(1)
                                randi_low = max([last_correct_read{k}(i) num_correct-max_grid_read_delay+1]);
                                c_read = randi([randi_low randi_high]);
                                last_correct_read{k}(i) = c_read;
                                u_read(i) = u_hist(i,c_read);
                            end
                        elseif (strcmp(async_type, 'semi-async') == 1)
                            %randi_low = num_correct-max_grid_read_delay+1;
                            randi_low = max([last_correct_read(k) num_correct-max_grid_read_delay+1]);
                            randi_high = num_correct+1;
                            c_read = randi([randi_low randi_high]);
                            u_read = u_hist(:,c_read);
                        else
                            u_read = u;
                        end
                    else
                        u_read = u;
                    end
                    e = b - A{1}*u_read;
                    
                    if (strcmp(mg_type, 'afacx') == 1)
                        for i = 2:k
                            e = R{i}*e;
                        end
                        Pq = speye(N(k));
                        for i = (k-1):-1:1
                            Pq = P{i}*Pq;
                        end
                        if (k == q)
                            e = A{k}\e;
                        else
                            f = e;
                            if (strcmp(smooth_type, 'Jacobi') == 1)
                                e = Jacobi(A{k+1}, zeros(N(k+1),1), R{k+1}*e, num_relax, 1);
                                e = Jacobi(A{k}, zeros(N(k),1), f - A{k}*(P{k}*e), num_relax, 1);
                            elseif (strcmp(smooth_type, 'wJacobi') == 1)
                                e = Jacobi(A{k+1}, zeros(N(k+1),1), R{k+1}*e, num_relax, omega);
                                e = Jacobi(A{k}, zeros(N(k),1), f - A{k}*(P{k}*e), num_relax, omega);
                            elseif (strcmp(smooth_type, 'async-Jacobi') == 1)
                                e = async_Jacobi(A{k+1}, zeros(N(k+1),1), R{k+1}*e, num_relax, num_relax, max_smooth_wait, max_smooth_read_delay, 1);
                                e = async_Jacobi(A{k}, zeros(N(k),1), f - A{k}*(P{k}*e), num_relax, num_relax, max_smooth_wait, max_smooth_read_delay, 1);
                            elseif (strcmp(smooth_type, 'async-wJacobi') == 1)
                                e = async_Jacobi(A{k+1}, zeros(N(k+1),1), R{k+1}*e, num_relax, num_relax, max_smooth_wait, max_smooth_read_delay, omega);
                                e = async_Jacobi(A{k}, zeros(N(k),1), f - A{k}*(P{k}*e), num_relax, num_relax, max_smooth_wait, max_smooth_read_delay, omega);
                            else
                                e = GS_lower(A{k+1}, zeros(N(k+1),1), R{k+1}*e, num_relax);
                                e = GS_lower(A{k}, zeros(N(k),1), f - A{k}*(P{k}*e), num_relax);
                            end
                        end
                        for i = (k-1):-1:1
                            e = P{i}*e;
                        end
                    else
                        for i = 2:k
                            e = R{i}*(G{i-1}*e);
                        end
                        if (k == q)
                            e = A{q}\e;
                        else
                            f = e;
                            if (strcmp(smooth_type, 'Jacobi') == 1)
                                e = Jacobi(A{k}, zeros(N(k),1), f, num_relax, 1);
                            elseif (strcmp(smooth_type, 'symm-Jacobi') == 1)
                                e = SymmJacobi(A{k}, zeros(N(k),1), f, num_relax, omega);
                            elseif (strcmp(smooth_type, 'wJacobi') == 1)
                                e = Jacobi(A{k}, zeros(N(k),1), f, num_relax, omega);
                            elseif (strcmp(smooth_type, 'par-Southwell') == 1)
                                e = ParSouthwell(A{k}, zeros(N(k),1), f, num_relax, omega);
                            elseif (strcmp(smooth_type, 'Southwell') == 1)
                                e = Southwell(A{k}, zeros(N(k),1), f, num_relax, omega);
                            elseif (strcmp(smooth_type, 'async-Jacobi') == 1)
                                e = async_Jacobi(A{k}, zeros(N(k),1), f, num_relax, num_relax, max_smooth_wait, max_smooth_read_delay, 1);
                            elseif (strcmp(smooth_type, 'async-wJacobi') == 1)
                                e = async_Jacobi(A{k}, zeros(N(k),1), f, num_relax, num_relax, max_smooth_wait, max_smooth_read_delay, omega);
                            else
                                e = GS_lower(A{k}, zeros(N(k),1), f, num_relax);
                            end
%                             e = omega*d{k}.*e;
%                             e = (2./(omega*d{k})).*e - omega*A{k}*e;
%                             e = omega*d{k}.*e;
                        end
                        for i = (k-1):-1:1
                            e = G{i}*(P{i}*e);
                        end
                    end
                    e_write{k} = e;
                else
                    e_write{k} = zeros(N(1),1);
                end
            end
            
            for k = c_ind
                u = u + e_write{k};
                u_hist = [u_hist u];
                num_correct = num_correct + 1;
                if (strcmp(async_type, 'semi-async') == 1)
                    last_correct_read(k) = num_correct;
                end
            end
            
            model_time = model_time + 1;
            
            for k = 1:q
                if (grid_time_count(k) < grid_wait(k))
                    grid_time_count(k) = grid_time_count(k) + 1;
                else
                    if (async_flag == 1)
                        grid_wait(k) = randi([0 max_grid_wait]);
                    end
                    grid_time_count(k) = 0;
                end
            end
            
            if (sum(grid_flag) == q)
                break;
            end
        end
        
        if (async_flag == 1)
            grid_wait_list(iter+1) = model_time - prev_model_time - 1;
        end
        
        r = (b - A{1}*u);
        iter = iter + 1;
        
        if ((async_flag == 0) && (iter < max_iter))
            if (grid_wait_list(iter+1) > -1)
                grid_wait = grid_wait_list(iter+1)*ones(q,1);
            else
                grid_wait = max(randi([0 max_grid_wait],q,1))*ones(q,1);
            end
        end
        
        solve_hist(iter+1,1) = iter;
        solve_hist(iter+1,2) = model_time;
        solve_hist(iter+1,3) = norm(r)/r0_norm;
        if (print_flag == 1)
            fprintf('%2d %2d %e\n', ...
                     iter, model_time, norm(r)/r0_norm);
        end
    end
end