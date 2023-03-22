clear
close all

t = datetime('now');
folder = '3.17';
save_path = "data_save/light_data_"+folder;
% save_path = "data_save/2.23";

%% Network parameters
ori_rate = 10e6;
rec_rate = 60e6;
rate_times = rec_rate/ori_rate;
related_num = 8;
h_order = rate_times*related_num;
add_zero = h_order/2;

data_num = 100;
split_num = 10;  % Cut a signal into split_num shares

inputSize = h_order;
numHiddenUnits = 60;
outputSize = rate_times;  % y=h*x+n;  y:(outputSize,m) h:(outputSize,inputSize) x:(inputSize,m)
maxEpochs = 2000;
miniBatchSize = 200;
LearnRateDropPeriod = 12;
LearnRateDropFactor = 0.1;
inilearningRate = 1e-2;

ver = 2;
bias = 0.3;
%%
fprintf("This is Threenonlinear network , single amp , ini learningRate = %e , min batch size = %d , DropPeriod = %d , DropFactor = %f  v%d \n",...
    inilearningRate,miniBatchSize,LearnRateDropPeriod,LearnRateDropFactor,ver);

% cal_nmse = @(hat,exp)10*log10(sum(sum((hat-exp).^2))/sum(sum(exp.^2)));
% clearvars -except snr save_path snr_begin snr_end inilearningRate inputSize ...
%                   numHiddenUnits outputSize maxEpochs miniBatchSize ...
%                   LearnRateDropPeriod LearnRateDropFactor cal_nmse
test_num = 0;
loop_begin = 2;
loop_end = 26;
loop_step = 1;
loop_num = (loop_end - loop_begin)/loop_step + 1 ;

amp_begin = 0.0015;
amp_norm = 0.03994;
nmse_all = zeros(1,loop_num);
save_amp = zeros(1,loop_num);
for loop = loop_begin: loop_step :loop_end
%%  Load data
    test_num = test_num + 1;
    load_path = save_path + "/data/10M/rand_bias"+bias+"/amp"+loop+"/mat";
    fprintf("load amp=%d \n",loop);
    load_data
    totalNum = data_num*split_num;
    trainNum = floor(totalNum*0.85);
    xTrain_tmp = x(1:trainNum);
    yTrain_tmp = y(1:trainNum);
    xTest_tmp = x(trainNum+1:end);
    yTest_tmp = y(trainNum+1:end);

%     xTrain_tmp = cellfun(@(cell1)(cell1*100*1.1^amp),xTrain_tmp,'UniformOutput',false);
%     xTest_tmp = cellfun(@(cell1)(cell1*100*1.1^amp),xTest_tmp,'UniformOutput',false);
    
    amp_loop = 32000*(amp_begin+(loop-1)*amp_norm);
    save_amp(test_num) = 10*log10(amp_loop^2);
    xTrain_tmp = cellfun(@(cell1)(cell1*amp_loop),xTrain_tmp,'UniformOutput',false);
    xTest_tmp = cellfun(@(cell1)(cell1*amp_loop),xTest_tmp,'UniformOutput',false);

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

    load_path = "data_save/light_data_3.10/data/10M/rand_bias0.3/";
    norm_mat = load(load_path+"/save_norm.mat");
    norm_names = fieldnames(norm_mat);
    norm_factor = gather(eval(strcat('norm_mat.',norm_names{1})));
    xTrain = cellfun(@(cell1)(cell1*norm_factor),xTrain,'UniformOutput',false);
    xTest = cellfun(@(cell1)(cell1*norm_factor),xTest,'UniformOutput',false);
    band_power = bandpower(xTrain{10});
%%  Reshape data
    for i = 1:numel(xTrain)
        xTrain{i} = toeplitz(xTrain{i}(inputSize:-1:1),xTrain{i}(inputSize:end));
        yTrain{i} = reshape(yTrain{i}(1:split_length*rate_times),outputSize,split_length);
        yTrain{i} = yTrain{i}(:,1:size(xTrain{i},2));
    end
    
    for j = 1:numel(xTest)
        xTest{j} = toeplitz(xTest{j}(inputSize:-1:1),xTest{j}(inputSize:end));
        yTest{j} = reshape(yTest{j}(1:split_length*rate_times),outputSize,split_length);
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
    looptime = 2;
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
    
        fprintf("amp = %d , already training %d times \n",loop,i);
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
    if loop == loop_begin
        save_parameter = fopen(savePath_txt+"/save_parameter.txt",'w');
        fprintf(save_parameter,"\n \n");
        fprintf(save_parameter," Threenonlinear ,\r\n ini learningRate = %e ,\r\n min batch size = %d , \r\n " ...
            ,inilearningRate, miniBatchSize);
        fprintf(save_parameter,"DropPeriod = %d , DropFactor = %f ,\r\n ",LearnRateDropPeriod, LearnRateDropFactor);
        fprintf(save_parameter,"amp begin = %d , amp end = %d , amp step = %d \r\n ",loop_begin, loop_end, loop_step);
        fprintf(save_parameter,"data_num = %d , split num = %d , train num = %d\r\n",data_num,split_num,trainNum);
        fprintf(save_parameter," validationFrequency is floor(size(xTrain{1},2)/miniBatchSize \n");
        fprintf(save_parameter," origin rate = %e , receive rate = %e \n",ori_rate,rec_rate);
        fprintf(save_parameter," H order = %d ,related num = %d \n",h_order,related_num);
        fprintf(save_parameter," Hidden Units = %d \n",numHiddenUnits);
        fprintf(save_parameter," Add zero num = %d \n",add_zero);
        fclose(save_parameter);

    end

    if loop == loop_end
        save_nmse_name = 'save_nmse';
        eval([save_nmse_name,'=nmse_all;']);
        save(savePath_mat+"/save_Nmse.mat",save_nmse_name);
    end 

    if loop == loop_begin
        save_Nmse = fopen(savePath_txt+"/save_Nmse.txt",'w');
        save_bandpower = fopen(savePath_txt+"/save_bandpower.txt",'w');
        save_amp_txt = fopen(savePath_txt+"/save_amp.txt",'w');
    else
        save_Nmse = fopen(savePath_txt+"/save_Nmse.txt",'a');
        save_bandpower = fopen(savePath_txt+"/save_bandpower.txt",'a');
        save_amp_txt = fopen(savePath_txt+"/save_amp.txt",'a');
    end
    fprintf(save_Nmse,"%f \n" , nmse_mean);
    fprintf(save_bandpower,"%f \n" , band_power);
    fprintf(save_amp_txt,"%f \n" , save_amp(i));
    fclose(save_Nmse);
    fclose(save_bandpower);
    fclose(save_amp_txt);

    fprintf("amp %d training end \n",loop);
end
fprintf(" \n Threenonlinear, ini learningRate = %e , min batch size = %d , DropPeriod = %d , DropFactor = %f , data_num = %d \n",...
    inilearningRate, miniBatchSize, LearnRateDropPeriod, LearnRateDropFactor, data_num);
    
