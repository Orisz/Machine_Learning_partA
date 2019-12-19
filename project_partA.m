% part 0 - PCA
clear all; close all; clc;
Data = load('BreastCancerData.mat');
y_cell = num2cell(Data.y); %mapcaplot expects Label to be cell array
number_of_0 = length(find(Data.y == 0));
number_of_1 = 569 - number_of_0;
mapcaplot((Data.X)',y_cell);

%% divide data to training set and test set

idx_0s = find(Data.y == 0);
idx_1s = find(Data.y == 1);

% divide data while perserving ratio
train_0 = round(0.8*number_of_0);
train_1 = round(0.8*number_of_1);

train_set = Data.X(:,idx_0s(1:train_0));
train_set = [train_set Data.X(:,idx_1s(1:train_1))];

test_set = Data.X(:,idx_0s(train_0+1:end));
test_set = [test_set Data.X(:,idx_1s(train_1+1:end))];
gt_test_set = [Data.y(idx_0s(train_0+1:end)); Data.y(idx_1s(train_1+1:end))]; %ground truth labels for test set (for error estimation later on)
gt_train_set = [Data.y(idx_0s(1:train_0)) ; Data.y(idx_1s(1:train_1))];
%% assignment 1 - K-means
clc;
%section b
%K=2 for our data
tic;
[ y_labels, sqr_err, K_values ] = K_Means(Data.X, 2);
K_meansTime = toc;
%%
num_of_tagged_wrongKNN = sum(abs((y_labels-1)-Data.y)); 
pca_result = pca(Data.X'); %get principal component coefficients
%project data on the first two component coefficients
component1 = Data.X'*pca_result(:,1);
component2 = Data.X'*pca_result(:,2);
figure(1);
subplot(2,1,1); %show classification using K-means
%for healthy (K-means labels for 0)
x1_0KNN = component1(find(y_labels == 1));
x2_0KNN = component2(find(y_labels == 1));
plot(x1_0KNN,x2_0KNN,'bo');
title('Classification using K-means');
xlabel('pca 1st component');
ylabel('pca 2nd component');
hold on;
%for sick (K-means labels for 1)
x1_1KNN = component1(find(y_labels == 2));
x2_1KNN = component2(find(y_labels == 2));
plot(x1_1KNN,x2_1KNN,'ro');
hold off;
subplot(2,1,2);%show ground truth classification 
%for healthy (y=0)
plot(component1(find(Data.y == 0)),component2(find(Data.y == 0)),'bo');
title('Ground Truth Classified Samples');
xlabel('pca 1st component');
ylabel('pca 2nd component');
hold on;
%for sick (y=1)
plot(component1(find(Data.y == 1)),component2(find(Data.y == 1)),'ro');
hold off;


%section d
[ y_labels4, sqr_err4, K_values4 ] = K_Means(Data.X, 4); %K=4

figure(1);
subplot(2,1,1); %show classification using K-means
%for K=1 
plot(component1(find(y_labels4 == 1)),component2(find(y_labels4 == 1)),'bo');
title('Classification using K-means');
xlabel('pca 1st component');
ylabel('pca 2nd component');
hold on;
%for K=2 
plot(component1(find(y_labels4 == 2)),component2(find(y_labels4 == 2)),'ro');
hold on;
%for K=3
plot(component1(find(y_labels4 == 3)),component2(find(y_labels4 == 3)),'go');
hold on;
%for K=4
plot(component1(find(y_labels4 == 4)),component2(find(y_labels4 == 4)),'ko');
hold off;
subplot(2,1,2);%show ground truth classification 
%for healthy (y=0)
plot(component1(find(Data.y == 0)),component2(find(Data.y == 0)),'bo');
title('Ground Truth Classified Samples');
xlabel('pca 1st component');
ylabel('pca 2nd component');
hold on;
%for sick (y=1)
plot(component1(find(Data.y == 1)),component2(find(Data.y == 1)),'ro');
hold off;


%%
tic;
%calc miu and cov-matrix for p(x|c1)
num_train_set = train_0 + train_1;

train_0_class = Data.X(:,idx_0s(1:train_0));
train_1_class = Data.X(:,idx_1s(1:train_1));
miu0 = sum(train_0_class,2)/train_0;
miu1 = sum(train_1_class,2)/train_1;

rep_miu0 = repmat(miu0,1,train_0);
rep_miu1 = repmat(miu1,1,train_1);

sigmaMat0 = sum((train_0_class-rep_miu0).^2,2)/train_0;
sigmaMat1 = sum((train_1_class-rep_miu1).^2,2)/train_1;

num_test = size(test_set,2);
Bayes_class = zeros(num_test,1);
Px_c0 = 0;
Px_c1 = 0;
Py0 = train_0/num_train_set;
Py1 = 1-Py0;
for i=1:num_test %for each new sample, decide using Bayes its classification
    Px_c0 = Py0*prod(normpdf(test_set(:,i),miu0,sqrt(sigmaMat0)));
    Px_c1 = Py1*prod(normpdf(test_set(:,i),miu1,sqrt(sigmaMat1)));
    
    if(Px_c0 < Px_c1)
        Bayes_class(i) = 1;
    end
end

classification_err = sum(abs(Bayes_class - gt_test_set))/num_test; %error for classification using Bayes

%calc train set error
Bayes_class_train = zeros(num_train_set,1);
Px_c0 = 0;
Px_c1 = 0;

for i=1:num_train_set %for each new sample, decide using Bayes its classification
    Px_c0 = Py0*prod(normpdf(train_set(:,i),miu0,sqrt(sigmaMat0)));
    Px_c1 = Py1*prod(normpdf(train_set(:,i),miu1,sqrt(sigmaMat1)));
    
    if(Px_c0 < Px_c1)
        Bayes_class_train(i) = 1;
    end
end

classification_err_train = sum(abs(Bayes_class_train - gt_train_set))/num_train_set; %error for classification using Bayes
elapsed_time = toc;

%% Assignment 3 - Logistic Regression

%randomize the samples used for train and test set
idx_0s = find(Data.y == 0);
idx_1s = find(Data.y == 1);

trainNum = round(0.8*length(Data.y));
trainNum0 = round(0.6*trainNum);
trainNum1 = round(0.4*trainNum);

trainSet = Data.X(:,idx_0s(1:trainNum0));
trainSet = [trainSet Data.X(:,idx_1s(1:trainNum1))];
GTtrainSet = Data.y(idx_0s(1:trainNum0));
GTtrainSet = [GTtrainSet; Data.y(idx_1s(1:trainNum1))];
trainRand = randperm(trainNum0+trainNum1);
trainSet = trainSet(:,trainRand);
GTtrainSet = GTtrainSet(trainRand);

testSet = Data.X(:,idx_0s(trainNum0+1:end));
testSet = [testSet Data.X(:,idx_1s(trainNum1+1:end))];
GTtestSet = Data.y(idx_0s(trainNum0+1:end));
GTtestSet = [GTtestSet; Data.y(idx_1s(trainNum1+1:end))];
testRand = randperm(569-(trainNum0+trainNum1));
testSet = testSet(:,testRand);
GTtestSet = GTtestSet(testRand);

%center and normalize all data
CNTrainSet = (trainSet - mean(trainSet,2))./max(trainSet,[],2);
CNTestSet = (testSet - mean(testSet,2))./max(testSet,[],2);
tic;
%serial version

learnRate = [0.05,0.1,0.7];
for i=1:length(learnRate)
[TrainErr, TestErr] = SerialLR(CNTrainSet,GTtrainSet,CNTestSet,GTtestSet,learnRate(i));
x = 1:length(TrainErr);
figure(i);
plot(x,TrainErr,'b',x,TestErr,'r');
title(['Error vs Iterations (learn rate: ' num2str(learnRate(i)) ') - Serial version']);
ylabel('Error');
xlabel('iteration');
ylim([0 0.3]);
legend('Train Error','Test Error','Location', 'best');
end

elapsed_timeSerialLR = toc;
tic;
%batch version
for i=1:length(learnRate)
[TrainErr, TestErr] = BatchLR(CNTrainSet,GTtrainSet,CNTestSet,GTtestSet,learnRate(i));
x = 1:length(TrainErr);
figure(i+3);
plot(x,TrainErr,'b',x,TestErr,'r');
title(['Errors vs Iterations (learn rate: ' num2str(learnRate(i)) ') - Batch version']);
ylabel('Error');
xlabel('iteration');
ylim([0 0.1]);
grid on;
legend('Train Error','Test Error','Location', 'best');
end

elapsed_timeBatchLR = toc;

%% Assignment 4 - Decision Tree
tic;

CriteriaType = 3; %CriteriaType: 1 = Label-Error , 2 = Gini-index , 3 = Entropy
TestSetError = zeros(1,3);
meanDepth = zeros(1,3);
TreeDepth = zeros(1,3);
meanErrors = zeros(1,3);
STDErrors = zeros(1,3);
for i=1:CriteriaType 
    
    [Tree, TreeDepth(i), optError, meanDepth(i), meanErrors(i), STDErrors(i)] = optTree(train_set, gt_train_set, i);
    
    %test chosen tree on testSet. calculate error(%)
    TestSetError(i) = runTree(test_set, gt_test_set, Tree);
end

elapsed_timeTree = toc;