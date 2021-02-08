trial_num=100

m_active = 1;% number of features
n_active = 0;% number of samples
l1_active = 0;    % number of singular causes
l2_active = 0;    % number of pair causes
p_active = 0;% non-sparsity of signal
z_active = 0;% non-sparsity of necessary confounders
q_active = 0;% rate of noise
sp_active = 0;% distribution mean of 1D Prior Score



root_directory = "/home/aliarab/scratch/sgd4/sim_data/data_params"

if n_active
    fprintf("running:\n")
    m = 1000;% number of features
    n = 500;% number of samples
    l1 = 0;    % number of singular causes
    l2 = 1;    % number of pair causes
    l3 = 0; % number of triplet causes
    p = 1/4;% non-sparsity of signal
    z = 3/4;% non-sparsity of necessary confounders
    q = 0.05;% rate of noise
    sp = 3/4;% distribution mean of 1D Prior Score
    f = 28; % number of functions used
    n_params=[125 250 500 1000]
    for i = 1: length(n_params)
        for trial = 1:trial_num
            output_dir=sprintf('%s/n/%d/%d/', root_directory, n_params(i), trial)
            synthesize(m,n_params(i),l1,l2,l3,p,z,q,sp,f, output_dir);
            fprintf("executed\n")
        end
    end
end



if p_active
    fprintf("initializing...\n")
    m = 1000;% number of features
    n = 500;% number of samples
    l1 = 0;    % number of singular causes
    l2 = 1;    % number of pair causes
    l3 = 0; % number of triplet causes
    p = 1/4;% non-sparsity of signal
    z = 3/4;% non-sparsity of necessary confounders
    q = 0.05;% rate of noise
    sp = 3/4;% distribution mean of 1D Prior Score
    f = 28; % number of functions used
    p_params= [0.1 0.2 0.3 0.5]
    params=p_params
    for i = 1: length(params)
        for trial=1:trial_num
            output_dir=sprintf('%s/p/%.2f/%d/', root_directory, params(i), trial)
            synthesize(m,n,l1,l2,l3,params(i),z,q, sp,f, output_dir);
            fprintf("executed\n")
        end
    end
end


if z_active
    fprintf("initializing...\n")
    m = 1000;% number of features
    n = 500;% number of samples
    l1 = 0;    % number of singular causes
    l2 = 1;    % number of pair causes
    l3 = 0; % number of triplet causes
    p = 1/4;% non-sparsity of signal
    z = 3/4;% non-sparsity of necessary confounders
    q = 0.05;% rate of noise
    sp = 3/4;% distribution mean of 1D Prior Score
    f = 28; % number of functions used
    z_params=[1/4 1]
    params=z_params
    for i = 1: length(params)
        for trial=1:trial_num
            output_dir=sprintf('%s/z/%.2f/%d/', root_directory, params(i), trial)
            %disp(output_dir)
            synthesize(m,n,l1,l2,l3,p,params(i),q,sp,f, output_dir);
        end
    end
end


if q_active
    fprintf("initializing...\n")
    m = 1000;% number of features
    n = 500;% number of samples
    l1 = 0;    % number of singular causes
    l2 = 1;    % number of pair causes
    l3 = 0; % number of triplet causes
    p = 1/4;% non-sparsity of signal
    z = 3/4;% non-sparsity of necessary confounders
    q = 0.05;% rate of noise
    sp = 3/4;% distribution mean of 1D Prior Score
    f = 28; % number of functions used
    q_params =[0.025 0.05 0.1 0.2]
    params=q_params
    for i = 1: length(params)
        for trial = 1:trial_num
            output_dir=sprintf('%s/q/%.2f/%d/', root_directory, params(i), trial)
            synthesize(m,n,l1,l2,l3,p,z,q_params(i),sp,f, output_dir);
        end
    end
end


if sp_active
    fprintf("initializing...\n")
    m = 1000;% number of features
    n = 500;% number of samples
    l1 = 0;    % number of singular causes
    l2 = 1;    % number of pair causes
    l3 = 0; % number of triplet causes
    p = 1/4;% non-sparsity of signal
    z = 3/4;% non-sparsity of necessary confounders
    q = 0.05;% rate of noise
    sp = 3/4;% distribution mean of 1D Prior Score
    f = 28; % number of functions used
    S1p_params =[1/2 3/4 7/8 15/16]
    params=S1p_params
    for i = 1: length(params)
        for trial=1:trial_num
            output_dir=sprintf('%s/sp/%.2f/%d/', root_directory, params(i), trial)
            synthesize(m,n,l1,l2,l3,p,z,q,params(i),f, output_dir);
        end
    end
end


if l1_active
    fprintf("initializing...\n")
    m = 1000;% number of features
    n = 500;% number of samples
    l1 = 0;    % number of singular causes
    l2 = 1;    % number of pair causes
    l3 = 0; % number of triplet causes
    p = 1/4;% non-sparsity of signal
    z = 3/4;% non-sparsity of necessary confounders
    q = 0.05;% rate of noise
    sp = 3/4;% distribution mean of 1D Prior Score
    f = 28; % number of functions used
    l1_params =[0 1 2 3]
    params=l1_params
    for i = 1: length(params)
        for trial=1:trial_num
            output_dir=sprintf('%s/l1/%d/%d/', root_directory,  params(i), trial)
            synthesize(m,n,params(i),l2,l3,p,z,q,sp,f, output_dir);
        end
    end
end


if l2_active
    fprintf("initializing...\n")
    m = 1000;% number of features
    n = 500;% number of samples
    l1 = 0;    % number of singular causes
    l2 = 1;    % number of pair causes
    l3 = 0; % number of triplet causes
    p = 1/4;% non-sparsity of signal
    z = 3/4;% non-sparsity of necessary confounders
    q = 0.05;% rate of noise
    sp = 3/4;% distribution mean of 1D Prior Score
    f = 28; % number of functions used
    l2_params =[1 2 3 4]
    params=l2_params
    for i = 1: length(params)
        for trial=1:trial_num
            output_dir=sprintf('%s/l2/%d/%d/', root_directory, params(i), trial)
            synthesize(m,n,l1,params(i),l3,p,z,q,sp,f, output_dir);
        end
    end
end



if m_active
    fprintf("initializing...\n")
    m = 1000;% number of features
    n = 500;% number of samples
    l1 = 0;    % number of singular causes
    l2 = 1;    % number of pair causes
    l3 = 0; % number of triplet causes
    p = 1/4;% non-sparsity of signal
    z = 3/4;% non-sparsity of necessary confounders
    q = 0.05;% rate of noise
    sp = 3/4;% distribution mean of 1D Prior Score
    f = 28; % number of functions used
    m_params=[20000]
    params=m_params
    for i = 1: length(m_params)
        for trial=1:trial_num
            output_dir=sprintf('%s/m/%d/%d/', root_directory, params(i), trial)
            synthesize(params(i),n,l1,l2,l3,p,z,q, sp, f, output_dir);
            fprintf("executed\n")
        end
    end
    %exit;
end
