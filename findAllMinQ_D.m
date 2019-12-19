function [DTSIdx,DNum,thresh] = findAllMinQ_D(dividedTrainSet, dividedTrainLabel, DsOrigIdx, CriteriaType)
%findMinQ_D will call another function ("calcThreshMin" - finds the thresh
%for a specific D that will get the minimal criteria error)
%findMinQ_D finds the feature that gets Q(D) to minumum
%DsOrigIdx - to keep track of where each feature is in the tree
%DTSIdx - the row index of the chosen feature in the current dividedTrainSet

[Ds,N] = size(dividedTrainSet);

%criteriaMat: matrix Ds_numberx2 where rows indicate feature number.
%             col=1 is the Q(D) value given by the 
%             feature and col=2 is the corresponding threshold
criteriaMat = zeros(Ds,2);

for i=1:Ds
    [minQ_D,minThresh] = calcMinQ_D(dividedTrainSet(i,:),dividedTrainLabel,CriteriaType);
    criteriaMat(i,:) = [minQ_D,minThresh];
end

DTSIdx = find(criteriaMat(:,1) == min(criteriaMat(:,1))); %find the idx of minimal Q(D) value
DTSIdx = DTSIdx(1);
DNum = DsOrigIdx(DTSIdx);
thresh = criteriaMat(DTSIdx,2);
end

