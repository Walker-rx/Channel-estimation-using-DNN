% clear
% close all

% h_order = 6;
% xTest = cell(1,10);
% yTest = cell(1,10);
% xTest2 = cell(1,10);
% yTest2 = cell(1,10);
% 
% for i = 1:10
%     xTest{i} = 1:1200;
%     yTest{i} = 1:1200;
% end
% 
% for i = 1:numel(xTest)
%     xTest_p1 = toeplitz(xTest{i}(h_order:-1:1),xTest{i}(h_order:end));
%     xTest_p2 = toeplitz([0,xTest{i}(length(xTest{i})...
%         :-1:...
%         length(xTest{i})-(h_order-2)...
%         )...
%         ],...
%         zeros(1,h_order-1));
%     xTest2{i} = [xTest_p1,xTest_p2];
%     yTest2{i} = reshape(yTest{i},6,[]);
% end
clearvars -except net
amp = 41;
h_order = 30;
miniBatchSize = 40;
outputSize = 6;
save_path = "data_save/light_data";
load_path = save_path + "/25M/8pam/amp"+amp+"/mat";
fprintf("amp=%d \n",amp);
load_data
xTest = cellfun(@(cell1)(cell1*100*1.1^amp),x,'UniformOutput',false);
% yTest = cellfun(@(cell1)(cell1*100*1.1^amp),y,'UniformOutput',false);
for i = 1:numel(xTest)
    xTest{i} = toeplitz(xTest{i}(h_order:-1:1),xTest{i}(h_order:end))*100*1.1^amp;
    yTest{i} = reshape(yTest{i}(1:6000),outputSize,1000);
    yTest{i} = yTest{i}(:,1:size(xTest{i},2));
end
y_hat = predict(net,xTest,...
    'MiniBatchSize',miniBatchSize);
y_hatT = y_hat.';
mseNum = cellfun(@(cell1,cell2)mse(cell1,cell2),y_hatT,yTest);
Mse = mean(mseNum)

