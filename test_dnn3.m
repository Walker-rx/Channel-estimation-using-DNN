clear
% close all

t = datetime('now');
save_path = "data_save/light_data_2.28";
% save_path = "data_save/2.23";

%% Network parameters
h_order = 30;
inputSize = h_order;
numHiddenUnits = 25;
outputSize = 6;  % y=h*x+n;  y:(outputSize,m) h:(outputSize,inputSize) x:(inputSize,m)
maxEpochs = 2000;
miniBatchSize = 400;
LearnRateDropPeriod = 5;
LearnRateDropFactor = 0.1;
inilearningRate = 1e-2;

%%
fprintf("This is twononlinear network , ini learningRate = %e , min batch size = %d , DropPeriod = %d , DropFactor = %f  v3 \n",...
    inilearningRate,miniBatchSize,LearnRateDropPeriod,LearnRateDropFactor);

% cal_nmse = @(hat,exp)10*log10(sum(sum((hat-exp).^2))/sum(sum(exp.^2)));
% clearvars -except snr save_path snr_begin snr_end inilearningRate inputSize ...
%                   numHiddenUnits outputSize maxEpochs miniBatchSize ...
%                   LearnRateDropPeriod LearnRateDropFactor cal_nmse

%% Load data
test_num = 0;
amp_begin = -4;
amp_end = 50;
amp_step = 2;
for amp = amp_begin: amp_step :amp_end
    test_num = test_num + 1;
    load_path = save_path + "/data/25M/8pam/amp"+amp+"/mat";
    fprintf("load amp=%d \n",amp);
    load_data
    totalNum = data_num*10;
    trainNum = floor(totalNum*0.8);
    xTrain_tmp = x(1:trainNum);
    yTrain_tmp = y(1:trainNum);
    xTest_tmp = x(trainNum+1:end);
    yTest_tmp = y(trainNum+1:end);

    xTrain_tmp = cellfun(@(cell1)(cell1*100*1.1^amp),xTrain_tmp,'UniformOutput',false);
    xTest_tmp = cellfun(@(cell1)(cell1*100*1.1^amp),xTest_tmp,'UniformOutput',false);

    xTest_name = ['xTest',num2str(test_num)];
    yTest_name = ['yTest',num2str(test_num)];
    eval([xTest_name,'=xTest_tmp;']);
    eval([yTest_name,'=yTest_tmp;']);

    if amp == amp_begin
        xTrain = xTrain_tmp;
        yTrain = yTrain_tmp;
%             xTest = xTest_tmp;
%             yTest = yTest_tmp;
    else
        xTrain = [xTrain xTrain_tmp];
        yTrain = [yTrain yTrain_tmp];
%             xTest = [xTest xTest_tmp];
%             yTest = [yTest yTest_tmp];
    end
    clear x y
end
%     load_path = save_path + "/25M/8pam/amp"+amp+"/mat";
% %     load_path = save_path + "/data/snr"+snr;
%     fprintf("amp=%d \n",amp);
%     load_data
% 
%     totalNum = data_num*10;
%     trainNum = floor(totalNum*0.8);
%     xTrain1 = x(1:trainNum);
%     yTrain1 = y(1:trainNum);
%     xTest1 = x(trainNum+1:end);
%     yTest1 = y(trainNum+1:end);
%     xTrain1 = cellfun(@(cell1)(cell1*100*1.1^amp),xTrain1,'UniformOutput',false);
%     xTest1 = cellfun(@(cell1)(cell1*100*1.1^amp),xTest1,'UniformOutput',false);
%     clear x y
% 
%     load_path = save_path + "/25M/8pam/amp"+amp2+"/mat";
%     fprintf("amp=%d \n",amp2);
%     load_data
%     totalNum = data_num*10;
%     trainNum = floor(totalNum*0.8);
%     xTrain2 = x(1:trainNum);
%     yTrain2 = y(1:trainNum);
%     xTest2 = x(trainNum+1:end);
%     yTest2 = y(trainNum+1:end);
%     xTrain2 = cellfun(@(cell1)(cell1*100*1.1^amp2),xTrain2,'UniformOutput',false);
%     xTest2 = cellfun(@(cell1)(cell1*100*1.1^amp2),xTest2,'UniformOutput',false);
%     norm_factor = 1/norm(xTrain2{1})*sqrt(length(xTrain2{1}));
%     clear x y    
%  
%     load_path = save_path + "/25M/8pam/amp"+amp3+"/mat";
%     fprintf("amp=%d \n",amp3);
%     load_data
%     totalNum = data_num*10;
%     trainNum = floor(totalNum*0.8);
%     xTrain3 = x(1:trainNum);
%     yTrain3 = y(1:trainNum);
%     xTest3 = x(trainNum+1:end);
%     yTest3 = y(trainNum+1:end);
%     xTrain3 = cellfun(@(cell1)(cell1*100*1.1^amp3),xTrain3,'UniformOutput',false);
%     xTest3 = cellfun(@(cell1)(cell1*100*1.1^amp3),xTest3,'UniformOutput',false);
%     clear x y 

%%  Normalize data
totaltrain = numel(xTrain);
% norm_cell = xTrain{floor(totaltrain/2)};
% norm_factor = 1/norm(norm_cell)*sqrt(length(norm_cell));
load_path = save_path + "/result/3.1/25M/8pam/mix_amp/Twononlinear";
norm_mat = load(load_path+"/save_norm.mat");
norm_names = fieldnames(norm_mat);
norm_factor = gather(eval(strcat('norm_mat.',norm_names{1})));

xTrain = cellfun(@(cell1)(cell1*norm_factor),xTrain,'UniformOutput',false);

for i = 1:test_num
    xTest_nor = eval(['xTest',num2str(i)]);
    xTest_nor = cellfun(@(cell1)(cell1*norm_factor),xTest_nor,'UniformOutput',false);
    eval([['xTest',num2str(i)],'= xTest_nor;']);
end

%%  Reshape data
for i = 1:numel(xTrain)
    xTrain{i} = toeplitz(xTrain{i}(inputSize:-1:1),xTrain{i}(inputSize:end));
    yTrain{i} = reshape(yTrain{i}(1:6000),outputSize,1000);
    yTrain{i} = yTrain{i}(:,1:size(xTrain{i},2));
end
for i = 1:test_num
    xtop_tem = eval(['xTest',num2str(i)]);
    ytop_tem = eval(['yTest',num2str(i)]);
    for j = 1:numel(xtop_tem)            
        xtop_tem{j} = toeplitz(xtop_tem{j}(inputSize:-1:1),xtop_tem{j}(inputSize:end));            
        ytop_tem{j} = reshape(ytop_tem{j}(1:6000),outputSize,1000);
        ytop_tem{j} = ytop_tem{j}(:,1:size(xtop_tem{j},2));
%             xTest{i} = toeplitz(xTest{i}(inputSize:-1:1),xTest{i}(inputSize:end));
%             yTest{i} = reshape(yTest{i}(1:6000),outputSize,1000);
%             yTest{i} = yTest{i}(:,1:size(xTest1{i},2));
    end
    eval([['xTest',num2str(i)],'= xtop_tem;']);
    eval([['yTest',num2str(i)],'= ytop_tem;']);
end

%% Initialize network
validationFrequency = floor(size(xTrain{1},2)/miniBatchSize);

layers = [...
    sequenceInputLayer(inputSize)
    fullyConnectedLayer(numHiddenUnits)
    reluLayer
    fullyConnectedLayer(numHiddenUnits)
    reluLayer
    fullyConnectedLayer(outputSize)
    regressionLayer];
options = trainingOptions('adam', ...
    'GradientThreshold',1, ...
    'MaxEpochs',maxEpochs, ...
    'MiniBatchSize',miniBatchSize, ...
    'SequenceLength','longest', ...
    'Shuffle','every-epoch', ...
    'LearnRateSchedule','piecewise',...
    'LearnRateDropFactor',LearnRateDropFactor,...
    'LearnRateDropPeriod',LearnRateDropPeriod,...
    'ValidationData',{xTest1,yTest1},...
    'ValidationFrequency',validationFrequency,...
    'ValidationPatience',30,...
    'Verbose',true,...
    'InitialLearnRate',inilearningRate);
%         'Plots','training-progress');
% 'ExecutionEnvironment','gpu',...

%% Train network
looptime = 1;
for i = 1:test_num
    eval([['nmse',num2str(i),'_mat'],'= zeros(1,looptime);']);
end

for i = 1:looptime    
    net = trainNetwork(xTrain,yTrain,layers,options);
    h = net.Layers(2).Weights.';
    for j = 1:test_num
        x_fortest = eval(['xTest',num2str(j)]);
        y_fortest = eval(['yTest',num2str(j)]);
        y_hat = predict(net,x_fortest,'MiniBatchSize',miniBatchSize);
        y_hatT = y_hat.';

        nmseNum = cellfun(@(hat,exp)10*log10(sum(sum((hat-exp).^2))/sum(sum(exp.^2))),y_hatT,y_fortest);
        nmse_loop = mean(nmseNum);
        eval([['nmse',num2str(j),'_mat(',num2str(i),')'],'=nmse_loop;']);            
    end       
    fprintf("already training %d times \n",i);
end
nmse_mean = zeros(1,test_num);
for i = 1:test_num
    nmse_mean_tem = mean(eval(['nmse',num2str(i),'_mat']));
    nmse_mean(i) = nmse_mean_tem;
end

%% Save data
savePath_txt = save_path + "/result/"+t.Month+"."+t.Day+"/25M/8pam/Twononlinear3";   
savePath_mat = save_path + "/result/"+t.Month+"."+t.Day+"/25M/8pam/Twononlinear3"; 
if(~exist(savePath_txt,'dir'))
    mkdir(char(savePath_txt));
end
if(~exist(savePath_mat,'dir'))
    mkdir(char(savePath_mat));
end
save_parameter = fopen(savePath_txt+"/save_parameter.txt",'w');
fprintf(save_parameter,"\n \n");
fprintf(save_parameter," twononlinear ,\r\n ini learningRate = %e ,\r\n min batch size = %d , \r\n DropPeriod = %d ,\r\n DropFactor = %f ,\r\n amp begin = %d , amp end = %d , amp step = %d \r\n data_num = %d ",...
                                      inilearningRate, miniBatchSize, LearnRateDropPeriod, LearnRateDropFactor, amp_begin, amp_end, amp_step, data_num);
fclose(save_parameter);

save_nmse_name = 'save_nmse';
eval([save_nmse_name,'=nmse_mean;']);
save(savePath_mat+"/save_Nmse.mat",save_nmse_name);
for i = 1:test_num
%     save_nmse_name = ['save_nmse_' num2str(i)];
%     eval([save_nmse_name,'=nmse_mean(i);']);
    if i == 1
        save_Nmse = fopen(savePath_txt+"/save_Nmse.txt",'w');
%         save(savePath_mat+"/save_Nmse.mat",save_nmse_name);
    else
        save_Nmse = fopen(savePath_txt+"/save_Nmse.txt",'a');
%         save(savePath_mat+"/save_Nmse.mat",save_nmse_name,'-append');
    end
    fprintf(save_Nmse,"%f \n" , nmse_mean(i));
    fclose(save_Nmse);
end

fprintf(" \n twononlinear, ini learningRate = %e , min batch size = %d , DropPeriod = %d , DropFactor = %f , data_num = %d \n",...
    inilearningRate, miniBatchSize, LearnRateDropPeriod, LearnRateDropFactor, data_num);
    
