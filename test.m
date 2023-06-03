clear
% layers = [
%     imageInputLayer([28 28 1],'Name','input')
%     
%     convolution2dLayer(5,16,'Padding','same','Name','conv_1')
%     batchNormalizationLayer('Name','BN_1')
%     reluLayer('Name','relu_1')
%     
%     convolution2dLayer(3,32,'Padding','same','Stride',2,'Name','conv_2')
%     batchNormalizationLayer('Name','BN_2')
%     reluLayer('Name','relu_2')
%     convolution2dLayer(3,32,'Padding','same','Name','conv_3')
%     batchNormalizationLayer('Name','BN_3')
%     reluLayer('Name','relu_3')
%     
%     additionLayer(2,'Name','add')
%     
%     averagePooling2dLayer(2,'Stride',2,'Name','avpool')
%     fullyConnectedLayer(10,'Name','fc')
%     softmaxLayer('Name','softmax')
%     classificationLayer('Name','classOutput')];
% lgraph = layerGraph(layers);
% 
% skipConv = convolution2dLayer(1,32,'Stride',2,'Name','skipConv');
% lgraph = addLayers(lgraph,skipConv);
% lgraph = connectLayers(lgraph,'relu_1','skipConv');
% lgraph = connectLayers(lgraph,'skipConv','add/in2');
% figure
% plot(lgraph);
a=cell(1,5);
for i = 1:5
    a{i}=rand(5,10);
end
Train_cell_num = numel(a);
X = a;
X = cell2mat(X);
numOber = size(X,2);
idx = randperm(numOber);
X = X(:,idx);
X = mat2cell(X,size(X,1),repmat(size(X,2)/Train_cell_num,1,Train_cell_num));
b = X;
