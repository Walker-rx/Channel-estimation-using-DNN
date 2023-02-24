clear 
close all

save_path = "data_save/2.23";
snr_begin = -6;
snr_end = 42;
% snr = 2;
fprintf("This is Twononlinear network , v4 \n");
for snr = snr_begin:4:snr_end
    clearvars -except snr save_path snr_begin snr_end
    fprintf("snr = %d begin \n",snr);
    load_path = save_path + "/data/snr"+snr;
    load_data

    totalNum = data_num;
    trainNum = floor(totalNum*0.8);
    xTrain = x(1:trainNum);
    yTrain = y(1:trainNum);
    xTest = x(trainNum+1:end);
    yTest = y(trainNum+1:end);

    h_order = 30;
    inputSize = h_order;
    numHiddenUnits = 15;
    outputSize = 6;  % y=h*x+n;  y:(outputSize,m) h:(outputSize,inputSize) x:(inputSize,m)
    maxEpochs = 200;
    miniBatchSize = 40;

    for i = 1:numel(xTrain)
%         xTrain{i} = [zeros(15,1);xTrain{i}];
        xTrain{i} = toeplitz(xTrain{i}(h_order:-1:1),xTrain{i}(h_order:end));
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
    for i = 1:numel(xTest)
%         xTest{i} = [zeros(15,1);xTest{i}];
        xTest{i} = toeplitz(xTest{i}(h_order:-1:1),xTest{i}(h_order:end));
%         xTest_p2 = toeplitz([0,xTest{i}(length(xTest{i})...
%                                           :-1:...
%                                           length(xTest{i})-(h_order-2)...
%                                           ).'...
%                              ],...
%                              zeros(1,h_order-1));
%         xTest{i} = [xTest_p1,xTest_p2];
        yTest{i} = reshape(yTest{i}(1:6000),outputSize,1000);
        yTest{i} = yTest{i}(:,1:size(xTest{i},2));
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
        'LearnRateDropFactor',0.8,...
        'LearnRateDropPeriod',30,...
        'ValidationData',{xTest,yTest},...
        'ValidationFrequency',validationFrequency,...
        'ValidationPatience',5,...
        'Verbose',true);
%         'Plots','training-progress');
    % 'ExecutionEnvironment','gpu',...
%     pause(60)
    net = trainNetwork(xTrain,yTrain,layers,options);

    y_hat = predict(net,xTest,...
        'MiniBatchSize',miniBatchSize);
    y_hatT = y_hat.';

    mseNum = cellfun(@(cell1,cell2)mse(cell1,cell2),y_hatT,yTest);
    Mse = mean(mseNum);

    savePath_mat = save_path + "/result/Twononlinear/snr" + snr;
    savePath_txt = save_path + "/result/Twononlinear";
    if(~exist(savePath_mat,'dir'))
        mkdir(char(savePath_mat));
    end
    for j = 1:numel(y_hat)
        save_yHat = ['save_yHat_',num2str(j)];
        save_yTest = ['save_yTest_',num2str(j)];
        eval([save_yHat,'=y_hat{j};']);
        eval([save_yTest,'=yTest{j};']);
        if j == 1
            save(savePath_mat+"/save_yHat.mat",save_yHat);
            save(savePath_mat+"/save_yTest.mat",save_yTest);
        else
            save(savePath_mat+"/save_yHat.mat",save_yHat,'-append');
            save(savePath_mat+"/save_yTest.mat",save_yTest,'-append');
        end
        if mod(j,20) ==0
            progress = ['snr = ',num2str(snr),', save progress is ',num2str(j/numel(y_hat)*100),'%'];
            disp(progress);
        end
    end
    
    save_mseNum = 'saveMseNum';    
    eval([save_mseNum,'=mseNum;']);
    save(savePath_mat+"/save_mseNum.mat",save_mseNum);

    if snr == snr_begin
        save_snr = fopen(savePath_txt+"/save_snr.txt",'w');
        save_Mse = fopen(savePath_txt+"/save_Mse.txt",'w');
    else
        save_snr = fopen(savePath_txt+"/save_snr.txt",'a');
        save_Mse = fopen(savePath_txt+"/save_Mse.txt",'a');
    end 
    fprintf(save_snr,'%d \r\n',snr);
    fprintf(save_Mse,'%.6g \r\n',Mse);
    fclose(save_snr);
    fclose(save_Mse);
    fprintf(' snr = %d , mse = %.6g \r\n',snr,Mse);
    clearvars -except snr save_path snr_begin snr_end
end
