clear 
close all

save_path = "data_save/2.26";
snr_begin = 2;
snr_end = 50;
% snr = 2;
fprintf("This is linear network \n");
for snr = 50:4:snr_end
    clearvars -except snr save_path snr_begin snr_end
    fprintf("snr = %d begin \n",snr);
    load_path = save_path + "/data/snr"+snr;
    load_data

    totalNum = data_num;
    trainNum = floor(totalNum*0.8);

    amp = 33;
    x = cellfun(@(cell1)(cell1*100*1.1^amp),x,'UniformOutput',false);

    xTrain = x(1:trainNum).';
    yTrain = y(1:trainNum).';
    xTest = x(trainNum+1:end).';
    yTest = y(trainNum+1:end).';

    h_order = 20;
    inputSize = h_order;
    numHiddenUnits = 5;
    outputSize = 1;  % y=h*x+n;  y:(outputSize,m) h:(outputSize,inputSize) x:(inputSize,m)
    maxEpochs = 20000;
    miniBatchSize = 20;
    inilearningRate = 1e-4;
    fprintf("This is linear network , ini learningRate = %e ,v1 \n",inilearningRate);

    for i = 1:numel(xTrain)
        xTrain{i} = [zeros(1,10) xTrain{i}.'];
        xTrain{i} = toeplitz(xTrain{i}(h_order:-1:1),xTrain{i}(h_order:end));
        yTrain{i} = yTrain{i}.';
        yTrain{i} = yTrain{i}(:,1:size(xTrain{i},2));
    end

    for i = 1:numel(xTest)
        xTest{i} = [zeros(1,10) xTest{i}.'];
        xTest{i} = toeplitz(xTest{i}(h_order:-1:1),xTest{i}(h_order:end));
        yTest{i} = yTest{i}.';
        yTest{i} = yTest{i}(:,1:size(xTest{i},2));
    end

    validationFrequency = floor(size(xTrain{1},2)/miniBatchSize);

    layers = [...
        sequenceInputLayer(inputSize)
%         fullyConnectedLayer(numHiddenUnits)
%         reluLayer
        fullyConnectedLayer(outputSize)
        regressionLayer];
    options = trainingOptions('adam', ...
        'GradientThreshold',1, ...
        'MaxEpochs',maxEpochs, ...
        'MiniBatchSize',miniBatchSize, ...
        'SequenceLength','longest', ...
        'Shuffle','every-epoch', ...
        'LearnRateSchedule','piecewise',...
        'LearnRateDropFactor',0.1,...
        'LearnRateDropPeriod',60,...
        'ValidationData',{xTest,yTest},...
        'ValidationFrequency',validationFrequency,...
        'ValidationPatience',500,...
        'Verbose',true,...
        'InitialLearnRate',inilearningRate);
%         'Plots','training-progress');
    % 'ExecutionEnvironment','gpu',...
%     pause(60)

    looptime = 10;
    mmse = zeros(1,looptime);
    for k = 1:looptime    
        net = trainNetwork(xTrain,yTrain,layers,options);
        h = net.Layers(2).Weights.';
        y_hat = predict(net,xTest,...
            'MiniBatchSize',miniBatchSize);
        y_hatT = y_hat.';    
        mseNum = cellfun(@(cell1,cell2)mse(cell1,cell2),y_hatT,yTest);
        Mse = mean(mseNum);
        mmse(k) = Mse;
        fprintf("%d times , mse = %e \n",k,Mse);
    end
    Mmse = mean(mmse);
    fprintf("amp = %d , linear, ini learningRate = %e ,mse = %e \n",amp,inilearningRate,Mmse);

    savePath_mat = save_path + "/result/linear/snr" + snr;
    savePath_txt = save_path + "/result/linear";
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
    save_h = 'saveH'; 
    eval([save_mseNum,'=mseNum;']);    
    eval([save_h,'=h;']); 
    save(savePath_mat+"/save_mseNum.mat",save_mseNum);
    save(savePath_mat+"/save_h.mat",save_h);

    if snr == snr_begin
        save_snr = fopen(savePath_txt+"/save_snr.txt",'w');
        save_Mse = fopen(savePath_txt+"/save_Mse.txt",'w');
    else
        save_snr = fopen(savePath_txt+"/save_snr.txt",'a');
        save_Mse = fopen(savePath_txt+"/save_Mse.txt",'a');
    end
    fprintf(save_snr,'%d \r\n',snr);
    fprintf(save_Mse,'%.6g \r\n',Mmse);
    fclose(save_snr);
    fclose(save_Mse);
    fprintf(' snr = %d , mse = %.6g \r\n',snr,Mmse);
    clearvars -except snr save_path snr_begin snr_end
end
