clear 
close all

save_path = "data_save/line_data";
snr_begin = 2;
snr_end = 50;
% snr = 2;
for snr = 34:4:snr_end
    clearvars -except snr save_path snr_begin snr_end
    fprintf("snr = %d begin \n",snr);
    load_path = save_path + "/data/snr"+snr;
    load_data

    totalNum = data_num;
    trainNum = floor(totalNum/2);
    xTrain = x(1:trainNum);
    yTrain = y(1:trainNum);
    xTest = x(trainNum+1:end);
    yTest = y(trainNum+1:end);

    h_order = length(yTrain{1}) - length(xTrain{1}) + 1;
    inputSize = h_order;
    numHiddenUnits = 5;
    outputSize = 1;  % y = h*x+n; y:(1,n) h:(1,m) x:(m,n)
    maxEpochs = 40;
    miniBatchSize = 20;

    for i = 1:numel(xTrain)
        xTrain{i} = toeplitz(xTrain{i}(h_order:-1:1),xTrain{i}(h_order:end));
        xTest{i} = toeplitz(xTest{i}(h_order:-1:1),xTest{i}(h_order:end));
        yTrain{i} = yTrain{i}(h_order:length(yTrain{i})-(h_order-1)).';
        yTest{i} = yTest{i}(h_order:length(yTest{i})-(h_order-1)).';
    end
    validationFrequency = floor(size(xTrain{1},2)/miniBatchSize);

    layers = [...
        sequenceInputLayer(inputSize)
        fullyConnectedLayer(numHiddenUnits)
        fullyConnectedLayer(outputSize)
        regressionLayer];
    options = trainingOptions('adam', ...
        'GradientThreshold',1, ...
        'MaxEpochs',maxEpochs, ...
        'MiniBatchSize',miniBatchSize, ...
        'SequenceLength','longest', ...
        'Shuffle','every-epoch', ...
        'LearnRateSchedule','piecewise',...
        'LearnRateDropFactor',0.9,...
        'LearnRateDropPeriod',20,...
        'ValidationData',{xTest,yTest},...
        'ValidationFrequency',validationFrequency,...
        'ValidationPatience',5,...
        'Verbose',true, ...
        'Plots','training-progress');
    % 'ExecutionEnvironment','gpu',...
%     pause(60)
    net = trainNetwork(xTrain,yTrain,layers,options);

    y_hat = predict(net,xTest,...
        'MiniBatchSize',miniBatchSize);
    y_hatT = y_hat.';
    equalNum = cellfun(@(cell1,cell2)(abs(cell1-cell2)/cell2)<=1e-3,y_hatT,yTest);
    mseNum = cellfun(@(cell1,cell2)mse(cell1,cell2),y_hatT,yTest);
    Acc = mean(equalNum);
    Mse = mean(mseNum);

    savePath_mat = save_path + "/result/dnn2/snr" + snr;
    savePath_txt = save_path + "/result/dnn2";
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
    save_equalNum = 'saveEqualNum';
    save_mseNum = 'saveMseNum';
    eval([save_equalNum,'=equalNum;']);
    eval([save_mseNum,'=mseNum;']);
    save(savePath_mat+"/save_equalNum.mat",save_equalNum);
    save(savePath_mat+"/save_mseNum.mat",save_mseNum);

    if snr == snr_begin
        save_AccMse = fopen(savePath_txt+"/save_AccMse.txt",'w');
        save_snr = fopen(savePath_txt+"/save_snr.txt",'w');
        save_Mse = fopen(savePath_txt+"/save_Mse.txt",'w');
    else
        save_AccMse = fopen(savePath_txt+"/save_AccMse.txt",'a');
        save_snr = fopen(savePath_txt+"/save_snr.txt",'a');
        save_Mse = fopen(savePath_txt+"/save_Mse.txt",'a');
    end
    fprintf(save_AccMse,' snr = %d , acc = %f , mse = %.6g \r\n',snr,Acc,Mse);    
    fprintf(save_snr,'%d \r\n',snr);
    fprintf(save_Mse,'%.6g \r\n',Mse);
    fclose(save_AccMse);
    fclose(save_snr);
    fclose(save_Mse);
    fprintf(' snr = %d , acc = %f , mse = %.6g \r\n',snr,Acc,Mse);
    clearvars -except snr save_path snr_begin snr_end
%     clear x y xTrain yTrain xTest yTest y_hat y_hatT save_yHat save_yTest save_equalNum save_mseNum
%     clearvars -except snr save_path snr_begin snr_end
end
