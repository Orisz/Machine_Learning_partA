function [ y_labels, sqr_err_curr, K_values ] = K_Means(  data, K   )

data = data'; %so to have NxD for data
thresh = 0.01;
[N,D] = size(data);
K_mat = zeros(D,K);%the clusters' centers

sqr_err_prev = Inf;
diff_err_ratio = thresh + 10;

% unique_data_mat = unique(data,'rows'); %to make sure we don't pick the same cluster for some of the Ks
% random_indices = sort(randperm(length(unique_data_mat),K)); 
% K_mat = unique_data_mat(random_indices,:); %randomize clusters initialization from original data
K_mat = (data(randsample(N,K),:))';

K_mat = reshape(K_mat,1,D,K);

while (diff_err_ratio > thresh)
    %for each cluster selected find for each sample what is the
    %nearest cluster for him so we can later calc the new center of mass for each
    %cluster
    % 1st dim number of samples
    %2nd dim=D - number of features
    %3rd dim number of clusters (K)
    data_rep_3d = repmat(data,1,1 , K); % replicate data NxD matrix to NxDxK matrix
    K_mat_3d = repmat(K_mat,N,1,1); % replicate DxK clusters matrix to NxDxK
    Distance = sqrt(sum((data_rep_3d-K_mat_3d).^2,2));%2d NumOfSamples*K matrix of the distance of each sample's feature(each sample has D features) from each cluster
    D = reshape(Distance, N,K);
    [minDist , minIdx] = min(D,[],2);
    %where minIdx is the assignment of each sample to the closest cluster
    %calculate the square error
    sqr_err_curr = sum(minDist.^2)/N;
    diff_err_ratio = abs(sqr_err_curr - sqr_err_prev)/sqr_err_curr;
    sqr_err_prev = sqr_err_curr;
    
    %finding the new clusters' centers(K_matrix)
    for i=1:K
       sample_indices = find( minIdx==i);%what pixles are of level number i
       K_mat(1,:,i) = mean(data(sample_indices,:));
    end
end%end while

%get the final clusters' centers
K_values = reshape(K_mat,[],K);
y_labels = minIdx;

end%end func

