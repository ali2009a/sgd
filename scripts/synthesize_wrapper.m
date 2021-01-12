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
n_params=[125 250]
for i = 1: length(n_params)
    output_dir=sprintf('output/param_n_%d/',n_params(i))
    synthesize(m,n,l1,l2,l3,p,z,q,S1p,S2p,f, output_dir);
    fprintf("executed\n")
end
exit;
