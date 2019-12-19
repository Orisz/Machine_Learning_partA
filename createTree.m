function [Tree, depth] = createTree(Trainset, TrainLabels, CriteriaType)
%createTree: given the TrainData, create tree according to the rules we
%learned in the recitation
%Trainset: Dx(0.9)n - Train set
%TrainLabels: 1x(0.9)n - labels of train set people

%DsOrigIdx: 30x1 , for keeping track of original features' indices
DsOrigIdx = (1:30)';

%create tree root
nodes = []; %matrix where rows are numbered by the nodes' locations. cols=1 contains the feature number and cols=2 contains the feature's threshold
%createNode input: (nodeID according to tree structure, branch data, used features so far)
[nodes] = createNode(1,Trainset,TrainLabels,DsOrigIdx,nodes,CriteriaType);

Tree.nodes = nodes;
depth = floor(log2(size(Tree.nodes,1))); %depth of tree according to the last leaf


end

