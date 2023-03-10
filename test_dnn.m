clear
close all

t = datetime('now');
save_path = "data_save/light_data_3.10";
% save_path = "data_save/2.23";

%% Network parameters
h_order = 30;
inputSize = h_order;
numHiddenUnits = 40;
outputSize = 6;  % y=h*x+n;  y:(outputSize,m) h:(outputSize,inputSize) x:(inputSize,m)
maxEpochs = 2000;
miniBatchSize = 200;
LearnRateDropPeriod = 12;
LearnRateDropFactor = 0.1;
inilearningRate = 1e-2;
bias = 0.3;
ver = 1;
%%
fprintf("This is Threenonlinear network , single amp , ini learningRate = %e , min batch size = %d , DropPeriod = %d , DropFactor = %f  v%d \n",...
    inilearningRate,miniBatchSize,LearnRateDropPeriod,LearnRateDropFactor,ver);

% cal_nmse = @(hat,exp)10*log10(sum(sum((hat-exp).^2))/sum(sum(exp.^2)));
% clearvars -except snr save_path snr_begin snr_end inilearningRate inputSize ...
%                   numHiddenUnits outputSize maxEpochs miniBatchSize ...
%                   LearnRateDropPeriod LearnRateDropFactor cal_nmse
test_num = 0;
amp_begin = 1;
amp_end = 26;
amp_step = 1;
amp_num = (amp_end - amp_begin)/amp_step + 1 ;
nmse_all = zeros(1,amp_num);
for amp = amp_begin: amp_step :amp_end
%%  Load data
    test_num = test_num + 1;
    load_path = save_path + "/data/10M/rand_bias"+bias+"/amp"+amp+"/mat";
    fprintf("load amp=%d \n",amp);
    load_data
    totalNum = data_num*10;
    trainNum = floor(totalNum*0.8);
    xTrain_tmp = x(1:trainNum);
    yTrain_tmp = y(1:trainNum);
    xTest_tmp = x(trainNum+1:end);
    yTest_tmp = y(trainNum+1:end);

%     xTrain_tmp = cellfun(@(cell1)(cell1*100*1.1^amp),xTrain_tmp,'UniformOutput',false);
%     xTest_tmp = cellfun(@(cell1)(cell1*100*1.1^amp),xTest_tmp,'UniformOutput',false);

    xTrain_tmp = cellfun(@(cell1)(cell1*32000*(0.0015+(amp-1)*0.03994)),xTrain_tmp,'UniformOutput',false);
    xTest_tmp = cellfun(@(cell1)(cell1*32000*(0.0015+(amp-1)*0.03994)),xTest_tmp,'UniformOutput',false);

    xTrain = xTrain_tmp;
    yTrain = yTrain_tmp;
    xTest = xTest_tmp;
    yTest = yTest_tmp;

    clear x y

%%  Normalize data
    totaltrain = numel(xTrain);
    % norm_cell = xTrain{floor(totaltrain/2)};
    % norm_factor = 1/norm(norm_cell)*sqrt(length(norm_cell));
%     load_path = "data_save/light_data_2.28/result/3.1/25M/8pam/mix_amp/Twononlinear";

    load_path = "data_save/light_data_3.9/data/10M/rand_bias0.3/";
    norm_mat = load(load_path+"/save_norm.mat");
    norm_names = fieldnames(norm_mat);
    norm_factor = gather(eval(strcat('norm_mat.',norm_names{1})));
    xTrain = cellfun(@(cell1)(cell1*norm_factor),xTrain,'UniformOutput',false);
    xTest = cellfun(@(cell1)(cell1*norm_factor),xTest,'UniformOutput',false);

%%  Reshape data
    for i = 1:numel(xTrain)
        xTrain{i} = toeplitz(xTrain{i}(inputSize:-1:1),xTrain{i}(inputSize:end));
        yTrain{i} = reshape(yTrain{i}(1:6000),outputSize,1000);
        yTrain{i} = yTrain{i}(:,1:size(xTrain{i},2));
    end
    
    for j = 1:numel(xTest)
        xTest{j} = toeplitz(xTest{j}(inputSize:-1:1),xTest{j}(inputSize:end));
        yTest{j} = reshape(yTest{j}(1:6000),outputSize,1000);
        yTest{j} = yTest{j}(:,1:size(xTest{j},2));
    end

%% Initialize network
    validationFrequency = floor(size(xTrain{1},2)/miniBatchSize);
    
    layers = [...
        sequenceInputLayer(inputSize)
        fullyConnectedLayer(numHiddenUnits)
        reluLayer % 1
        fullyConnectedLayer(numHiddenUnits)
        reluLayer % 2
        fullyConnectedLayer(numHiddenUnits)
        reluLayer % 3
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
        'ValidationData',{xTest,yTest},...
        'ValidationFrequency',validationFrequency,...
        'ValidationPatience',30,...
        'Verbose',true,...
        'InitialLearnRate',inilearningRate);
    %         'Plots','training-progress');
    % 'ExecutionEnvironment','gpu',...

%% Train network
    looptime = 3;
    nmse_mat = zeros(1,looptime);
    
    for i = 1:looptime
        net = trainNetwork(xTrain,yTrain,layers,options);
        h = net.Layers(2).Weights.';
    
        x_fortest = xTest;
        y_fortest = yTest;
        y_hat = predict(net,x_fortest,'MiniBatchSize',miniBatchSize);
        y_hatT = y_hat.';
    
        nmseNum = cellfun(@(hat,exp)10*log10(sum(sum((hat-exp).^2))/sum(sum(exp.^2))),y_hatT,y_fortest);
        nmse_loop = mean(nmseNum);
        nmse_mat(i) = nmse_loop;
    
        fprintf("amp = %d , already training %d times \n",amp,i);
    end
    nmse_mean = mean(nmse_mat);
    nmse_all(test_num) = nmse_mean;
%% Save data
    savePath_txt = save_path + "/result/"+t.Month+"."+t.Day+"/10M/rand_bias"+bias+"/single_amp/Threenonlinear"+ver;
    savePath_mat = save_path + "/result/"+t.Month+"."+t.Day+"/10M/rand_bias"+bias+"/single_amp/Threenonlinear"+ver;
    if(~exist(savePath_txt,'dir'))
        mkdir(char(savePath_txt));
    end
    if(~exist(savePath_mat,'dir'))
        mkdir(char(savePath_mat));
    end
    if amp == amp_begin
        save_parameter = fopen(savePath_txt+"/save_parameter.txt",'w');
        fprintf(save_parameter,"\n \n");
        fprintf(save_parameter," Threenonlinear ,\r\n ini learningRate = %e ,\r\n min batch size = %d , \r\n DropPeriod = %d ,\r\n DropFactor = %f ,\r\n amp begin = %d , amp end = %d , amp step = %d \r\n data_num = %d \r\n",...
            inilearningRate, miniBatchSize, LearnRateDropPeriod, LearnRateDropFactor, amp_begin, amp_end, amp_step, data_num);
        fprintf(save_parameter," validationFrequency is floor(size(xTrain{1},2)/miniBatchSize");
        fprintf(save_parameter,"\n Hidden Units = %d",numHiddenUnits);
        fclose(save_parameter);
    end
    if amp == amp_end
        save_nmse_name = 'save_nmse';
        eval([save_nmse_name,'=nmse_all;']);
        save(savePath_mat+"/save_Nmse.mat",save_nmse_name);
    end    
    if amp == amp_begin
        save_Nmse = fopen(savePath_txt+"/save_Nmse.txt",'w');
    else
        save_Nmse = fopen(savePath_txt+"/save_Nmse.txt",'a');
    end
    fprintf(save_Nmse,"%f \n" , nmse_mean);
    fclose(save_Nmse);
    fprintf("amp %d training end \n",amp);
end
fprintf(" \n Threenonlinear, ini learningRate = %e , min batch size = %d , DropPeriod = %d , DropFactor = %f , data_num = %d \n",...
    inilearningRate, miniBatchSize, LearnRateDropPeriod, LearnRateDropFactor, data_num);
    
