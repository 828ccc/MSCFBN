%%MSCFBN
%%MS3DCN
clear all;
AAA_all10=[];
for times=1:1
load("Indian_pines_corrected.mat")
load("Indian_pines_gt.mat")
img=indian_pines_corrected;
gt=indian_pines_gt;
img_HSI=reshape(img,size(img,1)*size(img,2),size(img,3));%原始HSI数据200
img_label=reshape(gt,size(img,1)*size(img,2),1);
trainpercentage=5;
%% Dimensional Reduction and Normalization
Dims=60;
[img,~] = PCA_img(img, Dims);
sd = std(img,[],3);
img = img./sd; clear sd

%% Patchs
WS=9;
inputSize = [WS WS size(img,3)];
[allPatches,allLabels] = Create_Patches(img,gt,WS);
img1=reshape(img,145*145,size(img,3));%原始HSI数据15
%% Selecting Non-zero patches
patchesLabeled = allPatches(allLabels>0,:,:,:);    %% Total Patches of HSI
patchLabels = allLabels(allLabels>0);               %% Non Zero Labels
img1 = img1(allLabels>0,:);             %% Non Zero Labels
%%索引
[sequenceLengthsSorted,idx] = sort(patchLabels);
patchLabels=patchLabels(idx);
patchesLabeled=patchesLabeled(idx,:,:,:);
img1=img1(idx,:);
%%每类取trainpercentage=5个样本的索引
[Tr_Ind,Te_Ind]=index_5(patchLabels,trainpercentage);
img1_xtrain = img1(Tr_Ind,:);
img1_xtest = img1(Te_Ind,:);
img1_ytrain = patchLabels(Tr_Ind,:);
img1_ytest = patchLabels(Te_Ind,:);
%% Convert One-hot-encoded Labels to Categorical
patchLabels = categorical(patchLabels);
%% Disjoint Training and Test
    dataInputTrain = patchesLabeled(Tr_Ind,:,:,:);
    dataInputTrain = permute(dataInputTrain,[2 3 4 1]);        %% Training按照2341的顺序重新排列数组的维度
    TrC = patchLabels(Tr_Ind,1);
    Tr = augmentedImageDatastore(inputSize,dataInputTrain,TrC);
    
    dataInputTest = patchesLabeled(Te_Ind,:,:,:);
        TeC = patchLabels(Te_Ind,1);
        dataInputTest = permute(dataInputTest,[2 3 4 1]);      %% Test
        Te = augmentedImageDatastore(inputSize,dataInputTest,TeC);

%% Important Parameters
miniBatchSize = 256;        %% Mini Batch size to process in CNN
initLearningRate = 0.001;   %% Initial Learning Rate
learningRateFactor = 0.1;  %% Learning Rate
%numFilters = 64;
Epochs=100;
        uc=16;
numBlocks=2;

layer = [image3dInputLayer(inputSize,"Name","Input","Normalization","None")];
lgraph = layerGraph(layer);

outputName = "Input";
% outputName = layer.Name;

for i = 1:numBlocks
    dilationFactor = 2^(i-1);
    name_cov1="conv1_"+i;
    name_add="add_"+i;
       layers = [
            convolution3dLayer([3,3,3],60,"Name","conv2d_01"+i,Padding="same")
            reluLayer("Name","Relu_01"+i)
            convolution3dLayer([3,3,3],30,"Name","conv2d_02"+i,Padding="same")
            reluLayer("Name","Relu_02"+i)
            convolution3dLayer([3,3,3],10,"Name","conv2d_03"+i,Padding="same")
            reluLayer("Name","Relu_03"+i)
            concatenationLayer(3, 2, Name= "cat1"+i)
            concatenationLayer(3, 2, Name= "cat2"+i)
            ];
    lgraph = addLayers(lgraph,layers);

        layers = [
            convolution3dLayer([5,5,5],60,"Name","conv2d_11"+i,Padding="same")
            reluLayer("Name","Relu_11"+i)
            convolution3dLayer([5,5,5],30,"Name","conv2d_12"+i,Padding="same")
            reluLayer("Name","Relu_12"+i)
            convolution3dLayer([5,5,5],10,"Name","conv2d_13"+i,Padding="same")
            reluLayer("Name","Relu_13"+i)];
        lgraph = addLayers(lgraph,layers);

        layers = [
            convolution3dLayer([7,7,7],60,"Name","conv2d_21"+i,Padding="same")
            reluLayer("Name","Relu_21"+i)
            convolution3dLayer([7,7,7],30,"Name","conv2d_22"+i,Padding="same")
            reluLayer("Name","Relu_22"+i)
            convolution3dLayer([7,7,7],10,"Name","conv2d_23"+i,Padding="same")
            reluLayer("Name","Relu_23"+i)];
        lgraph = addLayers(lgraph,layers);

        lgraph = connectLayers(lgraph, outputName, "conv2d_01"+i);
        lgraph = connectLayers(lgraph, outputName, "conv2d_11"+i);
        lgraph = connectLayers(lgraph, outputName, "conv2d_21"+i);
        lgraph = connectLayers(lgraph, "Relu_13"+i, "cat1" + i + "/in2");
        lgraph = connectLayers(lgraph, "Relu_23"+i, "cat2" + i + "/in2");


    % Update layer output name.
    outputName = "cat2" + i;
end

layers = [
            fullyConnectedLayer(64,"Name","fc1")
            dropoutLayer(0.4,"Name","drop_1")
            fullyConnectedLayer(uc,"Name","fc2")
            softmaxLayer("Name","softmax")
            classificationLayer("Name","output")];
lgraph = addLayers(lgraph,layers);
lgraph = connectLayers(lgraph,outputName,"fc1");
   %% Specify Training Options
   options = trainingOptions("adam", "InitialLearnRate", initLearningRate,...
       "LearnRateSchedule","piecewise", "LearnRateDropPeriod", 1000, ...
            "LearnRateDropFactor", learningRateFactor, "MaxEpochs", Epochs, ...
                "MiniBatchSize", miniBatchSize, "GradientThresholdMethod",...
                    "l2norm", "GradientThreshold", 0.01, "VerboseFrequency"...
                        ,50, "ExecutionEnvironment", "auto");

[net, info] = trainNetwork(Tr, lgraph, options);
layer = "fc2";

YPrediction = classify(net,Te);
accuracy = sum(YPrediction == TeC)/numel(TeC)
featuresTrain = activations(net,Tr,layer,"OutputAs","rows");
featuresTest = activations(net,Te,layer,"OutputAs","rows");

save('MS3DCN_IP_fe.mat','featuresTrain','featuresTest','net');

end

%%MS2DCN
clear all;
AAA_all10=[];
% EPFresult1=[];
for times=1:1
load("Indian_pines_corrected.mat")
load("Indian_pines_gt.mat")
img=indian_pines_corrected;
gt=indian_pines_gt;
img_HSI=reshape(img,size(img,1)*size(img,2),size(img,3));%原始HSI数据200
img_label=reshape(gt,size(img,1)*size(img,2),1);
trainpercentage=5;
%% Dimensional Reduction and Normalization
Dims=60;
[img,~] = PCA_img(img, Dims);
sd = std(img,[],3);
img = img./sd; clear sd

%% Patchs
WS=9;
inputSize = [WS WS size(img,3)];
[allPatches,allLabels] = Create_Patches(img,gt,WS);
img1=reshape(img,145*145,size(img,3));%原始HSI数据15
%% Selecting Non-zero patches
patchesLabeled = allPatches(allLabels>0,:,:,:);    %% Total Patches of HSI
patchLabels = allLabels(allLabels>0);               %% Non Zero Labels
img1 = img1(allLabels>0,:);             %% Non Zero Labels
%%索引
[sequenceLengthsSorted,idx] = sort(patchLabels);
patchLabels=patchLabels(idx);
% for i=1:numel(idx)
%     patchesLabeled_sort(i,:,:,:)=patchesLabeled(idx(i),:,:,:);
% end
% patchesLabeled=patchesLabeled_sort;
patchesLabeled=patchesLabeled(idx,:,:,:);
img1=img1(idx,:);
%%每类取trainpercentage=5个样本的索引
[Tr_Ind,Te_Ind]=index_5(patchLabels,trainpercentage);
img1_xtrain = img1(Tr_Ind,:);
img1_xtest = img1(Te_Ind,:);
img1_ytrain = patchLabels(Tr_Ind,:);
img1_ytest = patchLabels(Te_Ind,:);
%% Convert One-hot-encoded Labels to Categorical
patchLabels = categorical(patchLabels);
%% Disjoint Training and Test
    dataInputTrain = patchesLabeled(Tr_Ind,:,:,:);
    dataInputTrain = permute(dataInputTrain,[2 3 4 1]);        %% Training按照2341的顺序重新排列数组的维度
    TrC = patchLabels(Tr_Ind,1);
    Tr = augmentedImageDatastore(inputSize,dataInputTrain,TrC);
    
    dataInputTest = patchesLabeled(Te_Ind,:,:,:);
        TeC = patchLabels(Te_Ind,1);
        dataInputTest = permute(dataInputTest,[2 3 4 1]);      %% Test
        Te = augmentedImageDatastore(inputSize,dataInputTest,TeC);

%% Important Parameters
miniBatchSize = 256;        %% Mini Batch size to process in CNN
initLearningRate = 0.001;   %% Initial Learning Rate
learningRateFactor = 0.1;  %% Learning Rate
%numFilters = 64;
Epochs=100;
        uc=16;
numBlocks=3;

layer = [imageInputLayer(inputSize,"Name","Input","Normalization","None")];
lgraph = layerGraph(layer);

outputName = "Input";
% outputName = layer.Name;

for i = 1:numBlocks
    dilationFactor = 2^(i-1);
    name_cov1="conv1_"+i;
    name_add="add_"+i;
       layers = [
            convolution2dLayer(3,60,"Name","conv2d_01"+i,Padding="same")
            reluLayer("Name","Relu_01"+i)
            convolution2dLayer(3,30,"Name","conv2d_02"+i,Padding="same")
            reluLayer("Name","Relu_02"+i)
            convolution2dLayer(3,10,"Name","conv2d_03"+i,Padding="same")
            reluLayer("Name","Relu_03"+i)
            concatenationLayer(3, 2, Name= "cat1"+i)
            concatenationLayer(3, 2, Name= "cat2"+i)
            ];
    lgraph = addLayers(lgraph,layers);

        layers = [
            convolution2dLayer(5,60,"Name","conv2d_11"+i,Padding="same")
            reluLayer("Name","Relu_11"+i)
            convolution2dLayer(5,30,"Name","conv2d_12"+i,Padding="same")
            reluLayer("Name","Relu_12"+i)
            convolution2dLayer(5,10,"Name","conv2d_13"+i,Padding="same")
            reluLayer("Name","Relu_13"+i)];
        lgraph = addLayers(lgraph,layers);

        layers = [
            convolution2dLayer(7,60,"Name","conv2d_21"+i,Padding="same")
            reluLayer("Name","Relu_21"+i)
            convolution2dLayer(7,30,"Name","conv2d_22"+i,Padding="same")
            reluLayer("Name","Relu_22"+i)
            convolution2dLayer(7,10,"Name","conv2d_23"+i,Padding="same")
            reluLayer("Name","Relu_23"+i)];
        lgraph = addLayers(lgraph,layers);

        lgraph = connectLayers(lgraph, outputName, "conv2d_01"+i);
        lgraph = connectLayers(lgraph, outputName, "conv2d_11"+i);
        lgraph = connectLayers(lgraph, outputName, "conv2d_21"+i);
        lgraph = connectLayers(lgraph, "Relu_13"+i, "cat1" + i + "/in2");
        lgraph = connectLayers(lgraph, "Relu_23"+i, "cat2" + i + "/in2");


    % Update layer output name.
    outputName = "cat2" + i;
end

layers = [
            fullyConnectedLayer(64,"Name","fc1")
            dropoutLayer(0.4,"Name","drop_1")
            fullyConnectedLayer(uc,"Name","fc2")
            softmaxLayer("Name","softmax")
            classificationLayer("Name","output")];
lgraph = addLayers(lgraph,layers);
lgraph = connectLayers(lgraph,outputName,"fc1");
   %% Specify Training Options
   options = trainingOptions("adam", "InitialLearnRate", initLearningRate,...
       "LearnRateSchedule","piecewise", "LearnRateDropPeriod", 1000, ...
            "LearnRateDropFactor", learningRateFactor, "MaxEpochs", Epochs, ...
                "MiniBatchSize", miniBatchSize, "GradientThresholdMethod",...
                    "l2norm", "GradientThreshold", 0.01, "VerboseFrequency"...
                        ,50, "ExecutionEnvironment", "auto");

[net, info] = trainNetwork(Tr, lgraph, options);
layer = "fc2";

YPrediction = classify(net,Te);
accuracy = sum(YPrediction == TeC)/numel(TeC)
featuresTrain1 = activations(net,Tr,layer,"OutputAs","rows");
featuresTest1 = activations(net,Te,layer,"OutputAs","rows");

load('MS3DCN_IP_fe.mat');

train_x=double(gather([featuresTrain,featuresTrain1]));
test_x=double(gather([featuresTest,featuresTest1]));
train_x=mapstd(train_x);
test_x=mapstd(test_x);

%%FBLS+guided filters
                        trainlen=length(img1_ytrain);
                        testlen=length(img1_ytest);
                        train_y=zeros(trainlen,uc);
                        test_y=zeros(testlen,uc);
                        
                        classkey=unique(img1_ytrain);
                        for i=1:length(classkey)
                            for j =1:trainlen
                                if img1_ytrain(j,1)==classkey(i,1)
                                    train_y(j,i)=1;
                                end    
                            end
                        end

                        for i=1:length(classkey)
                            for j =1:testlen
                                if img1_ytest(j,1)==classkey(i,1)
                                    test_y(j,i)=1;
                                end    
                            end
                        end
                        train_y=(train_y-1)*2+1;
                        test_y=(test_y-1)*2+1;
                

                      C = 2^-10;   %----C: the regularization parameter for sparse regualarization
                       s = .9;          %----s: 增强节点的收缩参数
                       best = 0.62;
                       result = [];
                       for NumRule=50                 %每个模糊子系统的模糊规则搜索范围
                          for NumFuzz=50             %模糊子系统个数搜索范围
                             for NumEnhan=50       %增强节点搜索范围
                                rand('state',1);
                                for i=1:NumFuzz
                                   alpha=rand(size(train_x,2),NumRule);
                                   Alpha{i}=alpha;
                                 end  %generating coefficients of the then part of fuzzy rules for each fuzzy system
                                 WeightEnhan=rand(NumFuzz*NumRule+1,NumEnhan); %%Iinitializing  weights connecting fuzzy subsystems  with enhancement layer
            
                                 fprintf(1, 'Fuzzy rule No.= %d, Fuzzy system. No. =%d, Enhan. No. = %d\n', NumRule, NumFuzz,NumEnhan);
                                 [NetoutTest,Training_time,Testing_time,TrainingAccuracy,TestingAccuracy,preditlabel]  = bls_train(train_x,train_y,test_x,test_y,Alpha,WeightEnhan,s,C,NumRule,NumFuzz);
                                 time =Training_time + Testing_time;
                                 result = [result; NumRule NumFuzz NumEnhan TrainingAccuracy TestingAccuracy];
                                 if best < TestingAccuracy
                                 best = TestingAccuracy;
                                  % save optimal.mat TrainingAccuracy TestingAccuracy  NumRule NumFuzz NumEnhan time
                                 end
                                 % clearvars -except best NumRule NumFuzz NumEnhan train_data train_label test_data test_label  s C result NetoutTest
                              end
                            end
                       end

                        para=1;
                       times=1;
                       H=size(img,1);
                       W=size(img,2);
                       max_d = max(img_HSI(:));
                        min_d = min(img_HSI(:));
                        img_HSI = (img_HSI-min_d)/(max_d-min_d);
                       blsOA(times,1)=TestingAccuracy;
                         predlabel=preditlabel;
                        % for i=1:9
                        %     for j=1: length(preditlabel)
                        %        if preditlabel(j,1)==i
                        %          predlabel(j,1)=Testkey(i,1);
                        %        end  
                        %     end
                        % end  %将分类后的标签从1-9转化成从1-16
                        % BLSresult=[predlabel' trainlabel'];
                        testlen=length(predlabel);
                        BLSresult=img_label;
                        for i=1:testlen
                            k=Te_Ind(i,1);
                            BLSresult(k,1)=predlabel(i,1);
                        end

                        BLSresult=reshape(BLSresult,H,W);
                        % data=reshape(data,H*W,B);
                %         tic
                        EPFresult = EPF(3,1,img_HSI,BLSresult);
                %         toc
                        %%% shows the computing time of EPF
                        EPFresult =reshape(EPFresult,[H*W 1]);
                        % EPFresulttest = EPFresult(testindexrand,:);
                        EPFresulttest=EPFresult(Te_Ind,1) ;
                        %%%% Evaluation the performance of the EPF
                        [EPFOA,AA,kappa,CA]=confusion(img1_ytest,EPFresulttest);
%                         CAA=CA';
%                        EPFCA(times,:)=CAA;
%                        EPFAA(times,1)=AA;
%                        EPFKA(times,1)=kappa;
                        OA(para,times)=EPFOA
                        AAA_all=[accuracy,blsOA,EPFOA,AA,kappa];
%                         EPFresult1=[EPFresult1,EPFresult];
                        AAA_all10=[AAA_all;AAA_all10];

                            
end        


% save EPFresult;
% save allLabels;
%% Patches
function [patbData,patbLabel] = Create_Patches(HSI, gt, WS)
%% Padding
padding = floor((WS-1)/2);
zeroPaddingPatb = padarray(HSI,[padding,padding],0,"both");
%% 
[r,c,b] = size(HSI);
patbData = zeros(r*c,WS,WS,b);
patbLabel = zeros(r*c,1);
zeroPaddedInput = size(zeroPaddingPatb);
patbIdx = 1;
for i = (padding + 1):(zeroPaddedInput(1) - padding)
    for j = (padding + 1):(zeroPaddedInput(2) - padding)
        patb = zeroPaddingPatb(i - padding:i + padding, j - padding: j + padding,:);
        patbData(patbIdx,:,:,:) = patb;
        patbLabel(patbIdx,1) = gt(i-padding,j-padding);
        patbIdx = patbIdx+1;
    end
end
end