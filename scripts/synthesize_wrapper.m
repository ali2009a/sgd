fprintf("running:\n")
m = 100;% number of features
n = 500;% number of samples
l1 = 0;    % number of singular causes
l2 = 1;    % number of pair causes
l3 = 0; % number of triplet causes
p = 1/4;% non-sparsity of signal
z = 3/4;% non-sparsity of necessary confounders
q = 0.05;% rate of noise
S1p = 3/4;% distribution mean of 1D Prior Score
S2p = 1/2;% distribution mean of 2D Prior Score
f = 28; % number of functions used
n_params=[125 250 500 1000]
for i = 1: length(n_params)
    output_dir=sprintf('output/param_n_%d/',n_params(i))
    synthesize(m,n_params(i),l1,l2,l3,p,z,q,S1p,S2p,f, output_dir);
    fprintf("executed\n")
end



fprintf("initializing...\n")
m = 100;% number of features
n = 500;% number of samples
l1 = 0;    % number of singular causes
l2 = 1;    % number of pair causes
l3 = 0; % number of triplet causes
p = 1/4;% non-sparsity of signal
z = 3/4;% non-sparsity of necessary confounders
q = 0.05;% rate of noise
S1p = 3/4;% distribution mean of 1D Prior Score
S2p = 1/2;% distribution mean of 2D Prior Score
f = 28; % number of functions used
m_params=[2500 5000 10000 20000]
for i = 1: length(m_params)
    output_dir=sprintf('output/param_m_%d/',m_params(i))
    synthesize(m_params(i),n,l1,l2,l3,p,z,q,S1p,S2p,f, output_dir);
    fprintf("executed\n")
end
%exit;




fprintf("initializing...\n")
m = 100;% number of features
n = 500;% number of samples
l1 = 0;    % number of singular causes
l2 = 1;    % number of pair causes
l3 = 0; % number of triplet causes
p = 1/4;% non-sparsity of signal
z = 3/4;% non-sparsity of necessary confounders
q = 0.05;% rate of noise
S1p = 3/4;% distribution mean of 1D Prior Score
S2p = 1/2;% distribution mean of 2D Prior Score
f = 28; % number of functions used
p_params= [1/16 1/8 1/4 1/2]
params=p_params
for i = 1: length(params)
    output_dir=sprintf('output/param_p_%.2f/',params(i))
    synthesize(m,n,l1,l2,l3,params(i),z,q,S1p,S2p,f, output_dir);
    fprintf("executed\n")
end



fprintf("initializing...\n")
m = 100;% number of features
n = 500;% number of samples
l1 = 0;    % number of singular causes
l2 = 1;    % number of pair causes
l3 = 0; % number of triplet causes
p = 1/4;% non-sparsity of signal
z = 3/4;% non-sparsity of necessary confounders
q = 0.05;% rate of noise
S1p = 3/4;% distribution mean of 1D Prior Score
S2p = 1/2;% distribution mean of 2D Prior Score
f = 28; % number of functions used
z_params=[1/2 3/4 7/8 15/16]
params=z_params
for i = 1: length(params)
    output_dir=sprintf('output/param_z_%.2f/',params(i))
    synthesize(m,n,l1,l2,l3,p,params(i),q,S1p,S2p,f, output_dir);
end


fprintf("initializing...\n")
m = 100;% number of features
n = 500;% number of samples
l1 = 0;    % number of singular causes
l2 = 1;    % number of pair causes
l3 = 0; % number of triplet causes
p = 1/4;% non-sparsity of signal
z = 3/4;% non-sparsity of necessary confounders
q = 0.05;% rate of noise
S1p = 3/4;% distribution mean of 1D Prior Score
S2p = 1/2;% distribution mean of 2D Prior Score
f = 28; % number of functions used
q_params =[0.025 0.05 0.1 0.2]
params=q_params
for i = 1: length(params)
    output_dir=sprintf('output/param_q_%.2f/',params(i))
    synthesize(m,n,l1,l2,l3,p,z,q_params(i),S1p,S2p,f, output_dir);
end


fprintf("initializing...\n")
m = 100;% number of features
n = 500;% number of samples
l1 = 0;    % number of singular causes
l2 = 1;    % number of pair causes
l3 = 0; % number of triplet causes
p = 1/4;% non-sparsity of signal
z = 3/4;% non-sparsity of necessary confounders
q = 0.05;% rate of noise
S1p = 3/4;% distribution mean of 1D Prior Score
S2p = 1/2;% distribution mean of 2D Prior Score
f = 28; % number of functions used
S1p_params =[1/2 3/4 7/8 15/16]
params=S1p_params
for i = 1: length(params)
    output_dir=sprintf('output/param_s1p_%.2f/',params(i))
    synthesize(m,n,l1,l2,l3,p,z,q,params(i),S2p,f, output_dir);
end



fprintf("initializing...\n")
m = 100;% number of features
n = 500;% number of samples
l1 = 0;    % number of singular causes
l2 = 1;    % number of pair causes
l3 = 0; % number of triplet causes
p = 1/4;% non-sparsity of signal
z = 3/4;% non-sparsity of necessary confounders
q = 0.05;% rate of noise
S1p = 3/4;% distribution mean of 1D Prior Score
S2p = 1/2;% distribution mean of 2D Prior Score
f = 28; % number of functions used
S2p_params =[1/2 3/4 7/8 15/16]
params=S2p_params
for i = 1: length(params)
    output_dir=sprintf('output/param_s2p_%.2f/',params(i))
    synthesize(m,n,l1,l2,l3,p,z,q,S1p,params(i),f, output_dir);
end




fprintf("initializing...\n")
m = 100;% number of features
n = 500;% number of samples
l1 = 0;    % number of singular causes
l2 = 1;    % number of pair causes
l3 = 0; % number of triplet causes
p = 1/4;% non-sparsity of signal
z = 3/4;% non-sparsity of necessary confounders
q = 0.05;% rate of noise
S1p = 3/4;% distribution mean of 1D Prior Score
S2p = 1/2;% distribution mean of 2D Prior Score
f = 28; % number of functions used
l1_params =[0 1 2 3]
params=l1_params
for i = 1: length(params)
    output_dir=sprintf('output/param_l1_%d/',params(i))
    synthesize(m,n,params(i),l2,l3,p,z,q,S1p,S2p,f, output_dir);
end


fprintf("initializing...\n")
m = 100;% number of features
n = 500;% number of samples
l1 = 0;    % number of singular causes
l2 = 1;    % number of pair causes
l3 = 0; % number of triplet causes
p = 1/4;% non-sparsity of signal
z = 3/4;% non-sparsity of necessary confounders
q = 0.05;% rate of noise
S1p = 3/4;% distribution mean of 1D Prior Score
S2p = 1/2;% distribution mean of 2D Prior Score
f = 28; % number of functions used
l2_params =[0 1 2 3]
params=l2_params
for i = 1: length(params)
    output_dir=sprintf('output/param_l2_%d/',params(i))
    synthesize(m,n,l1,params(i),l3,p,z,q,S1p,S2p,f, output_dir);
end
