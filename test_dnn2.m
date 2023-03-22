clear 
close all

save_path = "data_save/light_data";
% save_path = "data_save/2.23";

%% Network parameters
snr_begin = -6;
snr_end = 42;
h_order = 30;
inputSize = h_order;
numHiddenUnits = 25;
outputSize = 6;  % y=h*x+n;  y:(outputSize,m) h:(outputSize,inputSize) x:(inputSize,m)
maxEpochs = 2000;
miniBatchSize = 40;
LearnRateDropPeriod = 15;
LearnRateDropFactor = 0.1;
inilearningRate = 1e-2;

%%
snr = 34;
cal_nmse = @(hat,exp)10*log10(sum(sum((hat-exp).^2))/sum(sum(exp.^2)));
fprintf("This is twononlinear network , ini learningRate = %e , DropPeriod = %d , DropFactor = %f  v3 \n",...
        inilearningRate,LearnRateDropPeriod,LearnRateDropFactor);
% for snr = snr_begin:4:snr_end
    clearvars -except snr save_path snr_begin snr_end inilearningRate inputSize ...
                      numHiddenUnits outputSize maxEpochs miniBatchSize ...
                      LearnRateDropPeriod LearnRateDropFactor cal_nmse
%     fprintf("snr = %d begin \n",snr);
    amp = -1;
    amp2 = 33;
    amp3 = 41;

%% Load data
    load_path = save_path + "/25M/8pam/amp"+amp+"/mat";
%     load_path = save_path + "/data/snr"+snr;
    fprintf("amp=%d \n",amp);
    load_data

    totalNum = data_num*10;
    trainNum = floor(totalNum*0.7);
    xTrain1 = x(1:trainNum);
    yTrain1 = y(1:trainNum);
    xTest1 = x(trainNum+1:end);
    yTest1 = y(trainNum+1:end);
    xTrain1 = cellfun(@(cell1)(cell1*100*1.1^amp),xTrain1,'UniformOutput',false);
    xTest1 = cellfun(@(cell1)(cell1*100*1.1^amp),xTest1,'UniformOutput',false);
    clear x y

    load_path = save_path + "/25M/8pam/amp"+amp2+"/mat";
    fprintf("amp=%d \n",amp2);
    load_data
    totalNum = data_num*10;
    trainNum = floor(totalNum*0.8);
    xTrain2 = x(1:trainNum);
    yTrain2 = y(1:trainNum);
    xTest2 = x(trainNum+1:end);
    yTest2 = y(trainNum+1:end);
    xTrain2 = cellfun(@(cell1)(cell1*100*1.1^amp2),xTrain2,'UniformOutput',false);
    xTest2 = cellfun(@(cell1)(cell1*100*1.1^amp2),xTest2,'UniformOutput',false);
    norm_factor = 1/norm(xTrain2{1})*sqrt(length(xTrain2{1}));
    clear x y    
 
    load_path = save_path + "/25M/8pam/amp"+amp3+"/mat";
    fprintf("amp=%d \n",amp3);
    load_data
    totalNum = data_num*10;
    trainNum = floor(totalNum*0.8);
    xTrain3 = x(1:trainNum);
    yTrain3 = y(1:trainNum);
    xTest3 = x(trainNum+1:end);
    yTest3 = y(trainNum+1:end);
    xTrain3 = cellfun(@(cell1)(cell1*100*1.1^amp3),xTrain3,'UniformOutput',false);
    xTest3 = cellfun(@(cell1)(cell1*100*1.1^amp3),xTest3,'UniformOutput',false);
    clear x y 

%%  Normalize data
    xTrain = [xTrain1 xTrain2 xTrain3];
    yTrain = [yTrain1 yTrain2 yTrain3];
    xTrain = cellfun(@(cell1)(cell1*norm_factor),xTrain,'UniformOutput',false);
    xTest1 = cellfun(@(cell1)(cell1*norm_factor),xTest1,'UniformOutput',false);
    xTest2 = cellfun(@(cell1)(cell1*norm_factor),xTest2,'UniformOutput',false);
    xTest3 = cellfun(@(cell1)(cell1*norm_factor),xTest3,'UniformOutput',false);
  
%%  Reshape data
    for i = 1:numel(xTrain)
        xTrain{i} = toeplitz(xTrain{i}(inputSize:-1:1),xTrain{i}(inputSize:end));
        yTrain{i} = reshape(yTrain{i}(1:6000),outputSize,1000);
        yTrain{i} = yTrain{i}(:,1:size(xTrain{i},2));
    end
    for i = 1:numel(xTest1)
        xTest1{i} = toeplitz(xTest1{i}(inputSize:-1:1),xTest1{i}(inputSize:end));      
        yTest1{i} = reshape(yTest1{i}(1:6000),outputSize,1000);
        yTest1{i} = yTest1{i}(:,1:size(xTest1{i},2));
    end
    for i = 1:numel(xTest2)
        xTest2{i} = toeplitz(xTest2{i}(inputSize:-1:1),xTest2{i}(inputSize:end));
        yTest2{i} = reshape(yTest2{i}(1:6000),outputSize,1000);
        yTest2{i} = yTest2{i}(:,1:size(xTest2{i},2));
    end
    for i = 1:numel(xTest3)
        xTest3{i} = toeplitz(xTest3{i}(inputSize:-1:1),xTest3{i}(inputSize:end));
        yTest3{i} = reshape(yTest3{i}(1:6000),outputSize,1000);
        yTest3{i} = yTest3{i}(:,1:size(xTest3{i},2));
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
        'ValidationData',{xTest3,yTest3},...
        'ValidationFrequency',validationFrequency,...
        'ValidationPatience',30,...
        'Verbose',true,...
        'InitialLearnRate',inilearningRate);
%         'Plots','training-progress');
    % 'ExecutionEnvironment','gpu',...

%% Train network
    looptime = 10;
    nmse1_mat = zeros(1,looptime);
    nmse2_mat = zeros(1,looptime);
    nmse3_mat = zeros(1,looptime);
    for k = 1:looptime    
        net = trainNetwork(xTrain,yTrain,layers,options);
        h = net.Layers(2).Weights.';
        y_hat1 = predict(net,xTest1,...
            'MiniBatchSize',miniBatchSize);
        y_hatT1 = y_hat1.';
    
        y_hat2 = predict(net,xTest2,...
            'MiniBatchSize',miniBatchSize);
        y_hatT2 = y_hat2.';
    
        y_hat3 = predict(net,xTest3,...
            'MiniBatchSize',miniBatchSize);
        y_hatT3 = y_hat3.';
    
        nmseNum1 = cellfun(@(hat,exp)10*log10(sum(sum((hat-exp).^2))/sum(sum(exp.^2))),y_hatT1,yTest1);
        nmse1 = mean(nmseNum1);
        nmse1_mat(k) = nmse1;
    
        nmseNum2 = cellfun(@(hat,exp)10*log10(sum(sum((hat-exp).^2))/sum(sum(exp.^2))),y_hatT2,yTest2);
        nmse2 = mean(nmseNum2);
        nmse2_mat(k) = nmse2;
    
        nmseNum3 = cellfun(@(hat,exp)10*log10(sum(sum((hat-exp).^2))/sum(sum(exp.^2))),y_hatT3,yTest3);
        nmse3 = mean(nmseNum3);
        nmse3_mat(k) = nmse3;
        fprintf("%d times , amp nmse = %f , amp2 nmse = %f , amp3 nmse = %f \n",k,nmse1,nmse2,nmse3);
    end
    nmse1_mean = mean(nmse1_mat);
    nmse2_mean = mean(nmse2_mat);
    nmse3_mean = mean(nmse3_mat);
       
    savePath_txt = save_path + "/result/2.28/25M/Twononlinear";    
    if(~exist(savePath_txt,'dir'))
        mkdir(char(savePath_txt));
    end
    save_Nmse = fopen(savePath_txt+"/save_Nmse.txt",'w');
    fprintf(save_Nmse,"\n \n");
    fprintf(save_Nmse," twononlinear ,\r\n ini learningRate = %e ,\r\n DropPeriod = %d ,\r\n DropFactor = %f ,\r\n amp = %d , amp2 = %d , ampd = %f ,\r\n amp nmse = %f , amp2 nmse = %f , amp3 nmse = %f ",...
                                          inilearningRate,LearnRateDropPeriod,LearnRateDropFactor,amp,amp2,amp3,nmse1_mean,nmse2_mean,nmse3_mean);
    fclose(save_Nmse);

    fprintf(" \n twononlinear, ini learningRate = %e ,DropPeriod = %d , DropFactor = %f \n",inilearningRate,LearnRateDropPeriod,LearnRateDropFactor);
    fprintf(" amp nmse = %f , amp2 nmse = %f , amp3 nmse = %f \n",nmse1_mean,nmse2_mean,nmse3_mean);

%     savePath_mat = save_path + "/result/Twononlinear/snr" + snr;
%     savePath_txt = save_path + "/result/Twononlinear";
%     if(~exist(savePath_mat,'dir'))
%         mkdir(char(savePath_mat));
%     end
%     for j = 1:numel(y_hat)
%         save_yHat = ['save_yHat_',num2str(j)];
%         save_yTest = ['save_yTest_',num2str(j)];
%         eval([save_yHat,'=y_hat{j};']);
%         eval([save_yTest,'=yTest{j};']);
%         if j == 1
%             save(savePath_mat+"/save_yHat.mat",save_yHat);
%             save(savePath_mat+"/save_yTest.mat",save_yTest);
%         else
%             save(savePath_mat+"/save_yHat.mat",save_yHat,'-append');
%             save(savePath_mat+"/save_yTest.mat",save_yTest,'-append');
%         end
%         if mod(j,20) ==0
%             progress = ['snr = ',num2str(snr),', save progress is ',num2str(j/numel(y_hat)*100),'%'];
%             disp(progress);
%         end
%     end
%     
%     save_mseNum = 'saveMseNum';    
%     eval([save_mseNum,'=mseNum;']);
%     save(savePath_mat+"/save_mseNum.mat",save_mseNum);
% 
%     if snr == snr_begin
%         save_snr = fopen(savePath_txt+"/save_snr.txt",'w');
%         save_Mse = fopen(savePath_txt+"/save_Mse.txt",'w');
%     else
%         save_snr = fopen(savePath_txt+"/save_snr.txt",'a');
%         save_Mse = fopen(savePath_txt+"/save_Mse.txt",'a');
%     end 
%     fprintf(save_snr,'%d \r\n',snr);
%     fprintf(save_Mse,'%.6g \r\n',Mse);
%     fclose(save_snr);
%     fclose(save_Mse);
%     fprintf(' snr = %d , mse = %.6g \r\n',snr,Mse);
%     clearvars -except snr save_path snr_begin snr_end
% end
