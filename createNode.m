function [nodes] = createNode(nodeNum,dividedTrainSet,dividedTrainLabel,DsOrigIdx,nodes,CriteriaType)
%nodeNum: tells us where we are in the tree. node numbering convention
%dividedTrainSet: partial train set from parent node
%dividedTrainLabel: true labels of the partial train set from parent node
%nodes: matrix where cols are numbered by the nodes' locations. 
%      rows=1 contains the feature number and rows=2 contains the feature's 
%      threshold
%CriteriaType: 1 = Label-Error , 2 = Gini-index , 3 = Entropy


%stop condition: if dividedTrainSet that is received is tagged to the same
%class
if(length(unique(dividedTrainLabel)) == 1)
   nodes(nodeNum,1) = -1; %indicates a leaf
   nodes(nodeNum,2) = unique(dividedTrainLabel); %save tag of that leaf;
   return; 
end

[DTSIdx,DNum,thresh] = findAllMinQ_D(dividedTrainSet, dividedTrainLabel, DsOrigIdx, CriteriaType); %for each feature, find the minimal Q(D), where Q(D) is Label-Error/Gini-index/Entropy

%update nodes of tree
nodes(nodeNum,1) = DNum;
nodes(nodeNum,2) = thresh; %save the threshold (node decision)

leftData = dividedTrainSet(:,(dividedTrainSet(DTSIdx,:)<=nodes(nodeNum,2))); %for transfering data to left son
leftLabels = dividedTrainLabel((dividedTrainSet(DTSIdx,:)<=nodes(nodeNum,2))); %for transfering data labels to left son
leftData(DTSIdx,:) = []; %for left son not to use the feature that's already been used
nodeNumLeft = nodeNum*2; %node number of left son
rightData = dividedTrainSet(:,(dividedTrainSet(DTSIdx,:)>nodes(nodeNum,2))); %for transfering data to right son
rightLabels = dividedTrainLabel((dividedTrainSet(DTSIdx,:)>nodes(nodeNum,2))); %for transfering data labels to right son
rightData(DTSIdx,:) = []; %for right son not to use the feature that's already been used
nodeNumRight = nodeNum*2 + 1; %node number of right son
DsOrigIdx(DsOrigIdx == nodes(nodeNum,1)) = []; %delete number of feature used

%create left son
[nodes] = createNode(nodeNumLeft,leftData,leftLabels,DsOrigIdx,nodes,CriteriaType);

%create right son
[nodes] = createNode(nodeNumRight,rightData,rightLabels,DsOrigIdx,nodes,CriteriaType);

end

