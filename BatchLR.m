function [trainErr, testErr] = BatchLR(train_set,gt_train_set,test_set,gt_test_set,learnRate)
%SerialLR - calculate the optimal weights for training set and then test
%algorithm on test set. look for learnRate that gets the log-likelihood to
%minimum

w0 = rand(31,1); %initialize parameters
thresh = 0.000001;
[D, trN] = size(train_set);
maxIterations = 1300;
currItr = 0;
trainErr = [];
testErr = [];

%intialize vectors
w_t = w0;
w_tPlus1 = w0 + 1000;
currWD = 10;
while(currWD > thresh && currItr <= maxIterations)
    currItr = currItr + 1;

    deltaW = zeros(D+1,1);

    for j=1:trN %batch learning - in each iteration go over all train set
        curr_sample = [1;train_set(:,j)]; %get a new sample

        deltaW = deltaW + (gt_train_set(j) - phi(w_t'*curr_sample))*phi_der(w_t'*curr_sample)*curr_sample;
    end

    w_tPlus1 = w_t + learnRate*deltaW;

    trainErr = [trainErr , classError(train_set,gt_train_set,w_tPlus1)];
    testErr = [testErr , classError(test_set,gt_test_set,w_tPlus1)];

    %currWD = mean(abs(w_tPlus1 - w_t));
    currWD = norm(w_tPlus1 - w_t);
    w_t = w_tPlus1; %update w_tPlus1 for next iteration

end


end

function [y] = phi(v)
    y = 1/(1+exp(-v));
end

function [y] = phi_der(v)
    y = exp(-v)/((1+exp(-v))^2);
end

function [err] = classError(data_set,gt_data_set,w_tPlus1)
    
    [DData,N] = size(data_set);
    classTags = zeros(N,1);
    for idx=1:N
        y = phi(w_tPlus1'*[1;data_set(:,idx)]);
        if(y > 0.5)
            classTags(idx) = 1;
        end
    end
    
    err = sum((classTags - gt_data_set).^2)/N;
end

