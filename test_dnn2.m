clear 
close all

save_path = "data_save/light_data";
% save_path = "data_save/2.23";
snr_begin = -6;
snr_end = 42;
inilearningRate = 1e-2;
snr = 34;
fprintf("This is twononlinear network , ini learningRate = %e ,v3 \n",inilearningRate);
% for snr = snr_begin:4:snr_end
    clearvars -except snr save_path snr_begin snr_end inilearningRate
%     fprintf("snr = %d begin \n",snr);
    amp = -1;
    amp2 = 33;
    amp3 = 41;

    load_path = save_path + "/25M/8pam/amp"+amp+"/mat";
%     load_path = save_path + "/data/snr"+snr;
    fprintf("amp=%d \n",amp);
    load_data

%     x = cellfun(@(cell1)(cell1*100*1.1^amp),x,'UniformOutput',false);

%     totalNum = data_num;
%     trainNum = floor(totalNum*0.8);
%     xTrain = x(1:trainNum);
%     yTrain = y(1:trainNum);
%     xTest = x(trainNum+1:end);
%     yTest = y(trainNum+1:end);

    totalNum = data_num*10;
    trainNum = floor(totalNum*0.8);
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

    xTrain = [xTrain1 xTrain2 xTrain3];
    yTrain = [yTrain1 yTrain2 yTrain3];
    
    h_order = 30;
    inputSize = h_order;
    numHiddenUnits = 25;
    outputSize = 6;  % y=h*x+n;  y:(outputSize,m) h:(outputSize,inputSize) x:(inputSize,m)
    maxEpochs = 2000;
    miniBatchSize = 40;
    LearnRateDropPeriod = 10;
    LearnRateDropFactor = 0.1;
    
    for i = 1:numel(xTrain)
        xTrain{i} = toeplitz(xTrain{i}(h_order:-1:1),xTrain{i}(h_order:end));
        xTrain{i} = xTrain{i}/32e4*sqrt(numel(xTrain{i}));
%         xTrain_p2 = toeplitz([0,xTrain{i}(length(xTrain{i})...
%                                           :-1:...
%                                           length(xTrain{i})-(h_order-2)...
%                                           ).'...
%                              ],...
%                              zeros(1,h_order-1));
%         xTrain{i} = [xTrain_p1,xTrain_p2];
        yTrain{i} = reshape(yTrain{i}(1:6000),outputSize,1000);
        yTrain{i} = yTrain{i}(:,1:size(xTrain{i},2));
    end
    for i = 1:numel(xTest1)
        xTest1{i} = toeplitz(xTest1{i}(h_order:-1:1),xTest1{i}(h_order:end));
        xTest1{i} = xTest1{i}/32e4*sqrt(numel(xTest1{i}));
%         xTest_p2 = toeplitz([0,xTest{i}(length(xTest{i})...
%                                           :-1:...
%                                           length(xTest{i})-(h_order-2)...
%                                           ).'...
%                              ],...
%                              zeros(1,h_order-1));
%         xTest{i} = [xTest_p1,xTest_p2];
        yTest1{i} = reshape(yTest1{i}(1:6000),outputSize,1000);
        yTest1{i} = yTest1{i}(:,1:size(xTest1{i},2));
    end
    for i = 1:numel(xTest2)
        xTest2{i} = toeplitz(xTest2{i}(h_order:-1:1),xTest2{i}(h_order:end));
        xTest2{i} = xTest2{i}/32e4*sqrt(numel(xTest2{i}));
        yTest2{i} = reshape(yTest2{i}(1:6000),outputSize,1000);
        yTest2{i} = yTest2{i}(:,1:size(xTest2{i},2));
    end
    for i = 1:numel(xTest3)
        xTest3{i} = toeplitz(xTest3{i}(h_order:-1:1),xTest3{i}(h_order:end));
        xTest3{i} = xTest3{i}/32e4*sqrt(numel(xTest3{i}));
        yTest3{i} = reshape(yTest3{i}(1:6000),outputSize,1000);
        yTest3{i} = yTest3{i}(:,1:size(xTest3{i},2));
    end
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
%     pause(60)
looptime = 10;
mmse1 = zeros(1,looptime);
mmse2 = zeros(1,looptime);
mmse3 = zeros(1,looptime);
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

    mseNum1 = cellfun(@(cell1,cell2)mse(cell1,cell2),y_hatT1,yTest1);
    Mse1 = mean(mseNum1);
    mmse1(k) = Mse1;

    mseNum2 = cellfun(@(cell1,cell2)mse(cell1,cell2),y_hatT2,yTest2);
    Mse2 = mean(mseNum2);
    mmse2(k) = Mse2;

    mseNum3 = cellfun(@(cell1,cell2)mse(cell1,cell2),y_hatT3,yTest3);
    Mse3 = mean(mseNum3);
    mmse3(k) = Mse3;
    fprintf("%d times , amp mse = %e , amp2 mse = %e , amp3 mse = %e \n",k,Mse1,Mse2,Mse3);
end
    Mmse1 = mean(mmse1);
    Mmse2 = mean(mmse2);
    Mmse3 = mean(mmse3);
       
    savePath_txt = save_path + "/result/2.28/25M/Twononlinear";    
    if(~exist(savePath_txt,'dir'))
        mkdir(char(savePath_txt));
    end
    save_Mse = fopen(savePath_txt+"/save_Mse.txt",'a');
    fprintf(save_Mse,"\n \n");
    fprintf(save_Mse," twononlinear ,\r\n ini learningRate = %e ,\r\n DropPeriod = %d ,\r\n DropFactor = %f ,\r\n amp = %d , amp2 = %d , amp3 = %d ,\r\n amp mse = %e , amp2 mse = %e , amp3 mse = %e ",...
                                          inilearningRate,LearnRateDropPeriod,LearnRateDropFactor,amp,amp2,amp3,Mmse1,Mmse2,Mmse3);
    fclose(save_Mse);

    fprintf(" \n twononlinear, ini learningRate = %e ,DropPeriod = %d , DropFactor = %f \n",inilearningRate,LearnRateDropPeriod,LearnRateDropFactor);
    fprintf(" amp mse = %e , amp2 mse = %e , amp3 mse = %e \n",Mmse1,Mmse2,Mmse3);

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
