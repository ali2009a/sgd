function  synthesize(m,n,l1,l2,l3,p,z,q,Sp,f, output_dir)
%clc; clear variables; close all;
%m = 100;% number of features
%n = 500;% number of samples
%l1 = 0;	% number of singular causes
%l2 = 1;	% number of pair causes
%l3 = 0; % number of triplet causes
%p = 1/4;% non-sparsity of signal
%z = 3/4;% non-sparsity of necessary confounders
%q = 0.05;% rate of noise
%sp = 3/4;% distribution mean of 1D Prior Score
%f = 28; % number of functions used

disp(z)
%% Data Generation
X = rand(n,m); X = (X>1-p); y = zeros(n,1); % Features
v = datasample(1:m,1*l1+2*l2,'Replace',false); V = -ones(n,l1+l2); % Cause
for i = 1:l1 % Outcome
    x = X(:,v(i)); V(:,i) = x; y = y|x;
end
for i = 1:l2
    ii = l1+2*i; x = X(:,v(ii-1)).*X(:,v(ii)); V(:,l1+i) = x; y = y|x;
end
V1 = [v(1:l1)' v(1:l1)']; V2 = sort(reshape(v(l1+1:l1+2*l2),2,[])',2);
x = rand(n,1); x = (x>1-z); y = y&x; % Confounders
x = datasample(1:n,round(n*q),'Replace',false); y(x)=1-y(x); % Noise
S1 = 0.1*randn(1,m)+0.5; S1(S1>1)=1; S1(S1<0)=0; S1(v) = Sp; % 1D Prior
S2 = abs(corr(X)); % 2D Similarity
%disp([num2str(l1) ' single & ' num2str(l2) ' pair causes: '])
%disp(v)
%x = input('Export to CSV? [1/0]');
vv  = [V1;V2]
x=1;
mkdir(output_dir)
if x
    csvwrite(fullfile(output_dir,'W.csv'),S2); 
    csvwrite(fullfile(output_dir,'V.csv'),S1);
    csvwrite(fullfile(output_dir,'Y.csv'),y); 
    csvwrite(fullfile(output_dir,'X.csv'),X); 
    csvwrite(fullfile(output_dir,'true_causes_old.csv'),v)
    csvwrite(fullfile(output_dir,'true_causes.csv'),vv)
end

end
