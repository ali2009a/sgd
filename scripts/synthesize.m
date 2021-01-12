function  synthesize(m,n,l1,l2,l3,p,z,q,S1p,S2p,f, output_dir)
%clc; clear variables; close all;
%m = 100;% number of features
%n = 500;% number of samples
%l1 = 0;	% number of singular causes
%l2 = 1;	% number of pair causes
%l3 = 0; % number of triplet causes
%p = 1/4;% non-sparsity of signal
%z = 3/4;% non-sparsity of necessary confounders
%q = 0.05;% rate of noise
%S1p = 3/4;% distribution mean of 1D Prior Score
%S2p = 1/2;% distribution mean of 2D Prior Score
%f = 28; % number of functions used

%% Data Generation
% Creation of Features (p)
v = datasample(1:m,1*l1+2*l2+3*l3,'Replace',false);
X = rand(n,m); X = (X>1-p);
% Effect of Causes (l1&2&3)
V = -ones(n,l1+l2+l3); y = zeros(n,1);
for i = 1:l1
    x = X(:,v(i)); y = y|x;
    V(:,i) = x;
end
for i = 1:l2
    ii = l1+2*i; x = X(:,v(ii-1)).*X(:,v(ii));
    V(:,l1+i) = x; y = y|x;
end
for i = 1:l3
    ii = l1+2*l2+3*i;
    x = X(:,v(ii-2)).*X(:,v(ii-1)).*X(:,v(ii));
    V(:,l1+l2+i) = x; y = y|x;
end
V1 = [v(1:l1)' v(1:l1)'];
V2 = sort(reshape(v(l1+1:l1+2*l2),2,[])',2); %#ok<UDIM> %
% Effect of Confounders (z)
x = rand(n,1); x = (x>1-z); y = y&x;
% Effect of Noise (q)
x = datasample(1:n,round(n*q),'Replace',false); y(x)=1-y(x);
% Association: Correlation (XX)
XX = corr(X);
% 1D Prior
% px = truncate(makedist('Normal','mu',0.5,'sigma',0.1),0,1);
% pv = truncate(makedist('Normal','mu',S1p,'sigma',0.1),0,1);
% S1 = random(px,1,m); S1(v) = random(pv,size(v));
S1 = 0.1*randn(1,m)+0.5; S1(v) = 0*randn(size(v))+S1p;
S1(S1>1)=1; S1(S1<0)=0;
% 2D Prior
% S2 = zeros(m);
% for i = 1:m-1
% for j = i+1:m
% px = truncate(makedist('Normal','mu',XX(i,j),'sigma',0.1),0,1);
% S2(i,j) = random(px); S2(j,i) = S2(i,j);
% end
% end
% pv = truncate(makedist('Normal','mu',S2p,'sigma',0.1),0,1);
% for i = 1:l2
%     S2(V2(i,1),V2(i,2)) = random(pv);
% end
S2 = abs(0*randn(m)+XX);
for i = 1:l2
    S2(V2(i,1),V2(i,2)) = 0.1*randn+S2p;
    S2(V2(i,2),V2(i,1)) = S2(V2(i,1),V2(i,2));
end
S2(S2>1)=1; S2(S2<0)=0; S2 = tril(S2,-1)+tril(S2,-1)';

%% Export
disp(['Prevalences Y: %' num2str(100*nnz(y)/numel(y)) ...
    ', X: %' num2str(100*nnz(V)/numel(V))])
%x = input('Export to CSV? [1/0]');
x=1
mkdir(output_dir)
if x
    csvwrite('output/param_n_125/W.csv',S2)
    csvwrite(fullfile(output_dir,'W.csv'),S2); 
    csvwrite(fullfile(output_dir,'V.csv'),S1);
    csvwrite(fullfile(output_dir,'Y.csv'),y); 
    csvwrite(fullfile(output_dir,'X.csv'),X); 
    csvwrite(fullfile(output_dir,'true_causes.csv'),v)
end

end
