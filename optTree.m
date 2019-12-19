function [optTree, optDepth, optError, meanDepth, meanError, STDError] = optTree(TrainSet, TrainLabels, CriteriaType)
%OptTree: returns the optimal Tree that achieves the lowest mean error for train set
%configuration (cross-validation), according to the Criteria we're working with.

N = length(TrainLabels); % 80% of all data
setSize = floor(N/10);

%for choosing the optimal tree from the cross validation
currMinValidError = inf;
optTree = [];
optDepth = 0;
optError = 0;
meanDepth = 0;
ValidSetError = zeros(1,10); %10 sets of train set and valid set

%loop - cross validation for training. find the one to have the smallest error
%loo = leave one out (cross validation method)
for loo=1:10
    if(loo == 10)
        Train90 = TrainSet;
        Train90(:,((loo-1)*setSize+1):end) = [];
        TrainLabels90 = TrainLabels;
        TrainLabels90(((loo-1)*setSize+1):end) = []; 
        ValidSet = TrainSet(:,((loo-1)*setSize+1):end);
        ValidLabels = TrainLabels(((loo-1)*setSize+1):end);        
    else
        Train90 = TrainSet;
        Train90(:,((loo-1)*setSize+1):loo*setSize) = [];
        TrainLabels90 = TrainLabels;
        TrainLabels90(((loo-1)*setSize+1):loo*setSize) = []; 
        ValidSet = TrainSet(:,((loo-1)*setSize+1):loo*setSize);
        ValidLabels = TrainLabels(((loo-1)*setSize+1):loo*setSize);
    end

[Tree, depth] = createTree(Train90, TrainLabels90, CriteriaType);
meanDepth = meanDepth + depth;
%runTree on validation set
ValidSetError(loo) = runTree(ValidSet, ValidLabels, Tree);

 if(currMinValidError > ValidSetError(loo))
     currMinValidError = ValidSetError(loo);
     optTree = Tree;
     optDepth = depth;
     optError = currMinValidError;
 end

end
%end loo loop -------------------------------------------------------------

%calculate errors for all 10 validation leave-one-out set
meanDepth = meanDepth/10;
meanError = mean(ValidSetError);
STDError = std(ValidSetError);

end

