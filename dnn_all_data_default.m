clear
close all

t = datetime('now');
folder = '4.14';
save_path = "data_save/light_data_"+folder;

ver = 4;

savePath_txt = save_path + "/result1/"+t.Month+"."+t.Day+"/mix_bias_amp/Threenonlinear"+ver;   
savePath_mat = save_path + "/result1/"+t.Month+"."+t.Day+"/mix_bias_amp/Threenonlinear"+ver; 
if(~exist(savePath_txt,'dir'))
    mkdir(char(savePath_txt));
end
if(~exist(savePath_mat,'dir'))
    mkdir(char(savePath_mat));
end

%% Network parameters
ori_rate = 10e6;
rec_rate = 60e6;
rate_times = rec_rate/ori_rate;
related_num = 8;
h_order = rate_times*related_num;
add_zero = h_order/2;

split_num = 10;  % Cut a signal into split_num shares

inputSize = h_order+1;
numHiddenUnits = 60;
outputSize = rate_times;  % y=h*x+n;  y:(outputSize,m) h:(outputSize,inputSize) x:(inputSize,m)
maxEpochs = 2000;
LearnRateDropPeriod = 8;
LearnRateDropFactor = 0.1;
inilearningRate = 1e-2;


%%
fprintf("This is Threenonlinear network , ini learningRate = %e  , DropPeriod = %d , DropFactor = %f \n",...
    inilearningRate,LearnRateDropPeriod,LearnRateDropFactor);
fprintf("Hidden Units = %d , v%d \n",numHiddenUnits,ver)

train_loop_time = 1;
% bias_scope = 0.05:0.04:0.85;
% amp_scope_ini = [0.1613 0.32106 0.48082 0.64058 0.8003 1];
bias_scope = 0.85;
amp_scope_ini = 1;
data_scope = {[1 80] [81 160] [161 200] [201 280] [281 300]};
% data_scope = {[1 100]};
train_percent = 0.95;
total_data_num = 0;

dataset_order = cell(1,length(amp_scope_ini));
amp_data = [];
bias_data = cell(1,2);
train_time = 0;
amp_scope = amp_scope_ini;
while ~isempty(amp_scope)
    if length(amp_scope)>1
        amp_order = randperm(length(amp_scope),2);
        amp_data(1) = amp_scope(amp_order(1));
        amp_data(2) = amp_scope(amp_order(2));
        amp_scope(amp_order) = [];

        bias_scope_tmp = bias_scope;
        while length(bias_scope_tmp) >= 12

            miniBatchSize = 399;
            data = cell(1,2);
            %         data = cell(1,1);
            if  length(bias_scope_tmp) >=12 || length(bias_scope_tmp) <=21
                bias_order = randperm(length(bias_scope_tmp),floor(length(bias_scope_tmp)/2));
                bias_data{1} = bias_scope_tmp(bias_order);
                bias_scope_tmp(bias_order) = [];
                bias_data{2} = bias_scope_tmp;
            else
                bias_order = randperm(length(bias_scope_tmp),11);
                bias_data{1} = bias_scope_tmp(bias_order);
                bias_scope_tmp(bias_order) = [];
                bias_order = randperm(length(bias_scope_tmp),11);
                bias_data{2} = bias_scope_tmp(bias_order);
                bias_scope_tmp(bias_order) = [];
            end

            if length(bias_data{1})-length(bias_data{2})>0
                bias_data{2} = [bias_data{2} zeros(1,length(bias_data{1})-length(bias_data{2}))];
            elseif length(bias_data{1})-length(bias_data{2})<0
                bias_data{1} = [bias_data{1} zeros(1,length(bias_data{2})-length(bias_data{1}))];
            end
            data{1} = [amp_data(1) bias_data{1};...
                amp_data(2) bias_data{2}];
            data{2} = [amp_data(1) bias_data{2};...
                amp_data(2) bias_data{1}];

            amp_loop_data = [];
            bias_loop_data = cell(1,2);
            for load_scope = 1:numel(data_scope)
                load_begin = data_scope{load_scope}(1);
                load_end = data_scope{load_scope}(2);
                data_num = load_end-load_begin+1;
                total_data_num = total_data_num + data_num;
                for data_loop = 1:numel(data)
                    train_time = train_time+1;
                    load_bias_amp
                    %% Initialize network
                    validationFrequency = floor(numel(xTrain)/miniBatchSize/2);

                    layers = [...
                        sequenceInputLayer(inputSize)
                        fullyConnectedLayer(numHiddenUnits)
                        fullyConnectedLayer(numHiddenUnits)
                        reluLayer % 1
                        fullyConnectedLayer(numHiddenUnits)
                        reluLayer % 2
                        fullyConnectedLayer(numHiddenUnits)
                        sigmoidLayer % 3
                        %     fullyConnectedLayer(numHiddenUnits)
                        %     reluLayer % 4
                        %     fullyConnectedLayer(numHiddenUnits)
                        %     reluLayer % 5
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
                    for i = 1:test_num
                        eval([['nmse_valid',num2str(i),'_mat'],'= zeros(1,1);']);
                    end
                    xTrain_reshape = reshape(xTrain,[],test_num);
                    yTrain_reshape = reshape(yTrain,[],test_num);
                    for i = 1:test_num
                        xValid_i = ['xValid',num2str(i)];
                        eval([xValid_i ,'= xTrain_reshape(:,i);']);
                        yValid_i = ['yValid',num2str(i)];
                        eval([yValid_i ,'= yTrain_reshape(:,i);']);
                    end
                    if train_time<=1
                        net_path = savePath_mat+"/net/net"+train_time;
                        if(~exist(net_path,'dir'))
                            mkdir(char(net_path));
                        end

                        net = trainNetwork(xTrain,yTrain,layers,options);
                        save(net_path+"/net.mat",'net');  % Save the trained network

                        for j = 1:test_num
                            x_valid = eval(['xValid',num2str(j)]).';
                            y_valid = eval(['yValid',num2str(j)]).';

                            y_hat_valid = predict(net,x_valid,'MiniBatchSize',miniBatchSize);
                            y_hatT_valid = y_hat_valid.';

                            nmseNum_valid = cellfun(@(hat,exp)10*log10(sum(sum((hat-exp).^2))/sum(sum(exp.^2))),y_hatT_valid ,y_valid);
                            nmse_loop_valid = mean(nmseNum_valid);
                            eval([['nmse_valid',num2str(j),'_mat'],'=nmse_loop_valid;']);
                        end
                        fprintf("already training %d times \n",train_time);

                        nmse_valid_mean = zeros(1,test_num);
                        for i = 1:test_num
                            nmse_valid_mean_tem = mean(eval(['nmse_valid',num2str(i),'_mat']));
                            nmse_valid_mean(i) = nmse_valid_mean_tem;
                        end

                        for i = 1:test_num
                            if i == 1
                                save_Nmse_valid = fopen(net_path+"/save_Nmse_valid.txt",'w');
                                save_amp_bias_txt = fopen(net_path+"/save_amp.txt",'w');
                                %         save(savePath_mat+"/save_Nmse.mat",save_nmse_name);
                            else
                                save_Nmse_valid = fopen(net_path+"/save_Nmse_valid.txt",'a');
                                save_amp_bias_txt = fopen(net_path+"/save_amp.txt",'a');
                                %         save(savePath_mat+"/save_Nmse.mat",save_nmse_name,'-append');
                            end
                            fprintf(save_Nmse_valid,"%f \n" , nmse_valid_mean(i));
                            fprintf(save_amp_bias_txt," amp = %f , bias = %f ,bandpower = %f \n" , save_amp(i), bias_save(i), band_power(i));
                            if i == test_num
                                fprintf(save_amp_bias_txt," load begin = %d , load end = %d  \n" , load_begin,load_end);
                            end
                            fclose(save_Nmse_valid);
                            fclose(save_amp_bias_txt);
                        end

                        for i =1:test_num
                            eval(['clear xTest',num2str(i)])
                            eval(['clear yTest',num2str(i)])
                            eval(['clear xValid',num2str(i)])
                            eval(['clear yValid',num2str(i)])
                        end
                        clear xTrain yTrain
                        pause(10)
                    else
                        net_path = savePath_mat+"/net/net"+train_time;
                        if(~exist(net_path,'dir'))
                            mkdir(char(net_path));
                        end

                        load(savePath_mat+"/net/net"+(train_time-1)+"/net.mat");
                        layers = net.Layers;
                        net = trainNetwork(xTrain,yTrain,layers,options);
                        save(net_path+"/net.mat",'net');  % Save the trained network

                        for j = 1:test_num
                            x_valid = eval(['xValid',num2str(j)]).';
                            y_valid = eval(['yValid',num2str(j)]).';

                            y_hat_valid = predict(net,x_valid,'MiniBatchSize',miniBatchSize);
                            y_hatT_valid = y_hat_valid.';

                            nmseNum_valid = cellfun(@(hat,exp)10*log10(sum(sum((hat-exp).^2))/sum(sum(exp.^2))),y_hatT_valid ,y_valid);
                            nmse_loop_valid = mean(nmseNum_valid);
                            eval([['nmse_valid',num2str(j),'_mat(',num2str(i),')'],'=nmse_loop_valid;']);
                        end
                        fprintf("already training %d times \n",train_time);

                        nmse_valid_mean = zeros(1,test_num);
                        for i = 1:test_num
                            nmse_valid_mean_tem = mean(eval(['nmse_valid',num2str(i),'_mat']));
                            nmse_valid_mean(i) = nmse_valid_mean_tem;
                        end

                        for i = 1:test_num
                            if i == 1
                                save_Nmse_valid = fopen(net_path+"/save_Nmse_valid.txt",'w');
                                save_amp_bias_txt = fopen(net_path+"/save_amp.txt",'w');
                                %         save(savePath_mat+"/save_Nmse.mat",save_nmse_name);
                            else
                                save_Nmse_valid = fopen(net_path+"/save_Nmse_valid.txt",'a');
                                save_amp_bias_txt = fopen(net_path+"/save_amp.txt",'a');
                                %         save(savePath_mat+"/save_Nmse.mat",save_nmse_name,'-append');
                            end
                            fprintf(save_Nmse_valid,"%f \n" , nmse_valid_mean(i));
                            fprintf(save_amp_bias_txt," amp = %f , bias = %f ,bandpower = %f \n" , save_amp(i), bias_save(i), band_power(i));
                            if i == test_num
                                fprintf(save_amp_bias_txt," load begin = %d , load end = %d  \n" , load_begin,load_end);
                            end
                            fclose(save_Nmse_valid);
                            fclose(save_amp_bias_txt);
                        end

                        for i =1:test_num
                            eval(['clear xTest',num2str(i)])
                            eval(['clear yTest',num2str(i)])
                            eval(['clear xValid',num2str(i)])
                            eval(['clear yValid',num2str(i)])
                        end
                        clear xTrain yTrain
                        pause(10)
                    end
                end
            end

        end
        if ~isempty(bias_scope_tmp)

            data = cell(1,1);
            %         data = cell(1,1);
            bias_order = randperm(length(bias_scope_tmp));
            bias_data{1} = bias_scope_tmp(bias_order);
            bias_data{2} = bias_scope_tmp(bias_order);
            bias_scope_tmp(bias_order) = [];

            data{1} = [amp_data(1) bias_data{1};...
                amp_data(2) bias_data{2}];

            amp_loop_data = [];
            bias_loop_data = cell(1,size(data{1},1));
            for load_scope = 1:numel(data_scope)
                load_begin = data_scope{load_scope}(1);
                load_end = data_scope{load_scope}(2);
                data_num = load_end-load_begin+1;
                total_data_num = total_data_num + data_num;
                for data_loop = 1:numel(data)
                    train_time = train_time+1;
                    load_bias_amp
                    %% Initialize network                   
                    miniBatchSize = floor(numel(xTrain)/40);
                    validationFrequency = floor(numel(xTrain)/miniBatchSize/2);

                    layers = [...
                        sequenceInputLayer(inputSize)
                        fullyConnectedLayer(numHiddenUnits)
                        fullyConnectedLayer(numHiddenUnits)
                        reluLayer % 1
                        fullyConnectedLayer(numHiddenUnits)
                        reluLayer % 2
                        fullyConnectedLayer(numHiddenUnits)
                        sigmoidLayer % 3
                        %     fullyConnectedLayer(numHiddenUnits)
                        %     reluLayer % 4
                        %     fullyConnectedLayer(numHiddenUnits)
                        %     reluLayer % 5
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
                    for i = 1:test_num
                        eval([['nmse_valid',num2str(i),'_mat'],'= zeros(1,1);']);
                    end
                    xTrain_reshape = reshape(xTrain,[],test_num);
                    yTrain_reshape = reshape(yTrain,[],test_num);
                    for i = 1:test_num
                        xValid_i = ['xValid',num2str(i)];
                        eval([xValid_i ,'= xTrain_reshape(:,i);']);
                        yValid_i = ['yValid',num2str(i)];
                        eval([yValid_i ,'= yTrain_reshape(:,i);']);
                    end
                    if train_time<=1
                        net_path = savePath_mat+"/net/net"+train_time;
                        if(~exist(net_path,'dir'))
                            mkdir(char(net_path));
                        end

                        net = trainNetwork(xTrain,yTrain,layers,options);
                        save(net_path+"/net.mat",'net');  % Save the trained network

                        for j = 1:test_num
                            x_valid = eval(['xValid',num2str(j)]).';
                            y_valid = eval(['yValid',num2str(j)]).';

                            y_hat_valid = predict(net,x_valid,'MiniBatchSize',miniBatchSize);
                            y_hatT_valid = y_hat_valid.';

                            nmseNum_valid = cellfun(@(hat,exp)10*log10(sum(sum((hat-exp).^2))/sum(sum(exp.^2))),y_hatT_valid ,y_valid);
                            nmse_loop_valid = mean(nmseNum_valid);
                            eval([['nmse_valid',num2str(j),'_mat'],'=nmse_loop_valid;']);
                        end
                        fprintf("already training %d times \n",train_time);

                        nmse_valid_mean = zeros(1,test_num);
                        for i = 1:test_num
                            nmse_valid_mean_tem = mean(eval(['nmse_valid',num2str(i),'_mat']));
                            nmse_valid_mean(i) = nmse_valid_mean_tem;
                        end

                        for i = 1:test_num
                            if i == 1
                                save_Nmse_valid = fopen(net_path+"/save_Nmse_valid.txt",'w');
                                save_amp_bias_txt = fopen(net_path+"/save_amp.txt",'w');
                                %         save(savePath_mat+"/save_Nmse.mat",save_nmse_name);
                            else
                                save_Nmse_valid = fopen(net_path+"/save_Nmse_valid.txt",'a');
                                save_amp_bias_txt = fopen(net_path+"/save_amp.txt",'a');
                                %         save(savePath_mat+"/save_Nmse.mat",save_nmse_name,'-append');
                            end
                            fprintf(save_Nmse_valid,"%f \n" , nmse_valid_mean(i));
                            fprintf(save_amp_bias_txt," amp = %f , bias = %f ,bandpower = %f \n" , save_amp(i), bias_save(i), band_power(i));
                            if i == test_num
                                fprintf(save_amp_bias_txt," load begin = %d , load end = %d  \n" , load_begin,load_end);
                            end
                            fclose(save_Nmse_valid);
                            fclose(save_amp_bias_txt);
                        end

                        for i =1:test_num
                            eval(['clear xTest',num2str(i)])
                            eval(['clear yTest',num2str(i)])
                            eval(['clear xValid',num2str(i)])
                            eval(['clear yValid',num2str(i)])
                        end
                        clear xTrain yTrain
                        pause(10)
                    else
                        net_path = savePath_mat+"/net/net"+train_time;
                        if(~exist(net_path,'dir'))
                            mkdir(char(net_path));
                        end

                        load(savePath_mat+"/net/net"+(train_time-1)+"/net.mat");
                        layers = net.Layers;
                        net = trainNetwork(xTrain,yTrain,layers,options);
                        save(net_path+"/net.mat",'net');  % Save the trained network

                        for j = 1:test_num
                            x_valid = eval(['xValid',num2str(j)]).';
                            y_valid = eval(['yValid',num2str(j)]).';

                            y_hat_valid = predict(net,x_valid,'MiniBatchSize',miniBatchSize);
                            y_hatT_valid = y_hat_valid.';

                            nmseNum_valid = cellfun(@(hat,exp)10*log10(sum(sum((hat-exp).^2))/sum(sum(exp.^2))),y_hatT_valid ,y_valid);
                            nmse_loop_valid = mean(nmseNum_valid);
                            eval([['nmse_valid',num2str(j),'_mat(',num2str(i),')'],'=nmse_loop_valid;']);
                        end
                        fprintf("already training %d times \n",train_time);

                        nmse_valid_mean = zeros(1,test_num);
                        for i = 1:test_num
                            nmse_valid_mean_tem = mean(eval(['nmse_valid',num2str(i),'_mat']));
                            nmse_valid_mean(i) = nmse_valid_mean_tem;
                        end

                        for i = 1:test_num
                            if i == 1
                                save_Nmse_valid = fopen(net_path+"/save_Nmse_valid.txt",'w');
                                save_amp_bias_txt = fopen(net_path+"/save_amp.txt",'w');
                                %         save(savePath_mat+"/save_Nmse.mat",save_nmse_name);
                            else
                                save_Nmse_valid = fopen(net_path+"/save_Nmse_valid.txt",'a');
                                save_amp_bias_txt = fopen(net_path+"/save_amp.txt",'a');
                                %         save(savePath_mat+"/save_Nmse.mat",save_nmse_name,'-append');
                            end
                            fprintf(save_Nmse_valid,"%f \n" , nmse_valid_mean(i));
                            fprintf(save_amp_bias_txt," amp = %f , bias = %f ,bandpower = %f \n" , save_amp(i), bias_save(i), band_power(i));
                            if i == test_num
                                fprintf(save_amp_bias_txt," load begin = %d , load end = %d  \n" , load_begin,load_end);
                            end
                            fclose(save_Nmse_valid);
                            fclose(save_amp_bias_txt);
                        end

                        for i =1:test_num
                            eval(['clear xTest',num2str(i)])
                            eval(['clear yTest',num2str(i)])
                            eval(['clear xValid',num2str(i)])
                            eval(['clear yValid',num2str(i)])
                        end
                        clear xTrain yTrain
                        pause(10)
                    end
                end
            end
        end

    else

        amp_data(1) = amp_scope(1);
        amp_scope(1) = [];
        bias_scope_tmp = bias_scope;
        while length(bias_scope_tmp) >= 12
            miniBatchSize = 399;
            data = cell(1,1);
            if  length(bias_scope_tmp) >=12 || length(bias_scope_tmp) <=21
                bias_order = randperm(length(bias_scope_tmp),floor(length(bias_scope_tmp)/2));
                bias_data{1} = bias_scope_tmp(bias_order);
                bias_scope_tmp(bias_order) = [];
                bias_data{2} = bias_scope_tmp;
            else
                bias_order = randperm(length(bias_scope_tmp),11);
                bias_data{1} = bias_scope_tmp(bias_order);
                bias_scope_tmp(bias_order) = [];
                bias_order = randperm(length(bias_scope_tmp),11);
                bias_data{2} = bias_scope_tmp(bias_order);
                bias_scope_tmp(bias_order) = [];
            end

            if length(bias_data{1})-length(bias_data{2})>0
                bias_data{2} = [bias_data{2} zeros(1,length(bias_data{1})-length(bias_data{2}))];
            elseif length(bias_data{1})-length(bias_data{2})<0
                bias_data{1} = [bias_data{1} zeros(1,length(bias_data{2})-length(bias_data{1}))];
            end
            data{1} = [amp_data(1) bias_data{1};...
                amp_data(1) bias_data{2}];

            amp_loop_data = [];
            bias_loop_data = cell(1,size(data{1},1));
            for load_scope = 1:numel(data_scope)
                load_begin = data_scope{load_scope}(1);
                load_end = data_scope{load_scope}(2);
                data_num = load_end-load_begin+1;
                total_data_num = total_data_num + data_num;
                for data_loop = 1:numel(data)
                    train_time = train_time+1;
                    load_bias_amp
                    %% Initialize network
                    validationFrequency = floor(numel(xTrain)/miniBatchSize/2);

                    layers = [...
                        sequenceInputLayer(inputSize)
                        fullyConnectedLayer(numHiddenUnits)
                        fullyConnectedLayer(numHiddenUnits)
                        reluLayer % 1
                        fullyConnectedLayer(numHiddenUnits)
                        reluLayer % 2
                        fullyConnectedLayer(numHiddenUnits)
                        sigmoidLayer % 3
                        %     fullyConnectedLayer(numHiddenUnits)
                        %     reluLayer % 4
                        %     fullyConnectedLayer(numHiddenUnits)
                        %     reluLayer % 5
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
                    for i = 1:test_num
                        eval([['nmse_valid',num2str(i),'_mat'],'= zeros(1,1);']);
                    end
                    xTrain_reshape = reshape(xTrain,[],test_num);
                    yTrain_reshape = reshape(yTrain,[],test_num);
                    for i = 1:test_num
                        xValid_i = ['xValid',num2str(i)];
                        eval([xValid_i ,'= xTrain_reshape(:,i);']);
                        yValid_i = ['yValid',num2str(i)];
                        eval([yValid_i ,'= yTrain_reshape(:,i);']);
                    end
                    if train_time<=1
                        net_path = savePath_mat+"/net/net"+train_time;
                        if(~exist(net_path,'dir'))
                            mkdir(char(net_path));
                        end

                        net = trainNetwork(xTrain,yTrain,layers,options);
                        save(net_path+"/net.mat",'net');  % Save the trained network

                        for j = 1:test_num
                            x_valid = eval(['xValid',num2str(j)]).';
                            y_valid = eval(['yValid',num2str(j)]).';

                            y_hat_valid = predict(net,x_valid,'MiniBatchSize',miniBatchSize);
                            y_hatT_valid = y_hat_valid.';

                            nmseNum_valid = cellfun(@(hat,exp)10*log10(sum(sum((hat-exp).^2))/sum(sum(exp.^2))),y_hatT_valid ,y_valid);
                            nmse_loop_valid = mean(nmseNum_valid);
                            eval([['nmse_valid',num2str(j),'_mat'],'=nmse_loop_valid;']);
                        end
                        fprintf("already training %d times \n",train_time);

                        nmse_valid_mean = zeros(1,test_num);
                        for i = 1:test_num
                            nmse_valid_mean_tem = mean(eval(['nmse_valid',num2str(i),'_mat']));
                            nmse_valid_mean(i) = nmse_valid_mean_tem;
                        end

                        for i = 1:test_num
                            if i == 1
                                save_Nmse_valid = fopen(net_path+"/save_Nmse_valid.txt",'w');
                                save_amp_bias_txt = fopen(net_path+"/save_amp.txt",'w');
                                %         save(savePath_mat+"/save_Nmse.mat",save_nmse_name);
                            else
                                save_Nmse_valid = fopen(net_path+"/save_Nmse_valid.txt",'a');
                                save_amp_bias_txt = fopen(net_path+"/save_amp.txt",'a');
                                %         save(savePath_mat+"/save_Nmse.mat",save_nmse_name,'-append');
                            end
                            fprintf(save_Nmse_valid,"%f \n" , nmse_valid_mean(i));
                            fprintf(save_amp_bias_txt," amp = %f , bias = %f ,bandpower = %f \n" , save_amp(i), bias_save(i), band_power(i));
                            if i == test_num
                                fprintf(save_amp_bias_txt," load begin = %d , load end = %d  \n" , load_begin,load_end);
                            end
                            fclose(save_Nmse_valid);
                            fclose(save_amp_bias_txt);
                        end

                        for i =1:test_num
                            eval(['clear xTest',num2str(i)])
                            eval(['clear yTest',num2str(i)])
                            eval(['clear xValid',num2str(i)])
                            eval(['clear yValid',num2str(i)])
                        end
                        clear xTrain yTrain
                        pause(10)
                    else
                        net_path = savePath_mat+"/net/net"+train_time;
                        if(~exist(net_path,'dir'))
                            mkdir(char(net_path));
                        end

                        load(savePath_mat+"/net/net"+(train_time-1)+"/net.mat");
                        layers = net.Layers;
                        net = trainNetwork(xTrain,yTrain,layers,options);
                        save(net_path+"/net.mat",'net');  % Save the trained network

                        for j = 1:test_num
                            x_valid = eval(['xValid',num2str(j)]).';
                            y_valid = eval(['yValid',num2str(j)]).';

                            y_hat_valid = predict(net,x_valid,'MiniBatchSize',miniBatchSize);
                            y_hatT_valid = y_hat_valid.';

                            nmseNum_valid = cellfun(@(hat,exp)10*log10(sum(sum((hat-exp).^2))/sum(sum(exp.^2))),y_hatT_valid ,y_valid);
                            nmse_loop_valid = mean(nmseNum_valid);
                            eval([['nmse_valid',num2str(j),'_mat(',num2str(i),')'],'=nmse_loop_valid;']);
                        end
                        fprintf("already training %d times \n",train_time);

                        nmse_valid_mean = zeros(1,test_num);
                        for i = 1:test_num
                            nmse_valid_mean_tem = mean(eval(['nmse_valid',num2str(i),'_mat']));
                            nmse_valid_mean(i) = nmse_valid_mean_tem;
                        end

                        for i = 1:test_num
                            if i == 1
                                save_Nmse_valid = fopen(net_path+"/save_Nmse_valid.txt",'w');
                                save_amp_bias_txt = fopen(net_path+"/save_amp.txt",'w');
                                %         save(savePath_mat+"/save_Nmse.mat",save_nmse_name);
                            else
                                save_Nmse_valid = fopen(net_path+"/save_Nmse_valid.txt",'a');
                                save_amp_bias_txt = fopen(net_path+"/save_amp.txt",'a');
                                %         save(savePath_mat+"/save_Nmse.mat",save_nmse_name,'-append');
                            end
                            fprintf(save_Nmse_valid,"%f \n" , nmse_valid_mean(i));
                            fprintf(save_amp_bias_txt," amp = %f , bias = %f ,bandpower = %f \n" , save_amp(i), bias_save(i), band_power(i));
                            if i == test_num
                                fprintf(save_amp_bias_txt," load begin = %d , load end = %d  \n" , load_begin,load_end);
                            end
                            fclose(save_Nmse_valid);
                            fclose(save_amp_bias_txt);
                        end

                        for i =1:test_num
                            eval(['clear xTest',num2str(i)])
                            eval(['clear yTest',num2str(i)])
                            eval(['clear xValid',num2str(i)])
                            eval(['clear yValid',num2str(i)])
                        end
                        clear xTrain yTrain
                        pause(10)
                    end
                end
            end

        end
        if ~isempty(bias_scope_tmp)
            data = cell(1,1);
            %         data = cell(1,1);
            bias_order = randperm(length(bias_scope_tmp));
            bias_data{1} = bias_scope_tmp(bias_order);
            bias_data{2} = bias_scope_tmp(bias_order);
            bias_scope_tmp(bias_order) = [];

            data{1} = [amp_data(1) bias_data{1}];

            amp_loop_data = [];
            bias_loop_data = cell(1,size(data{1},1));
            for load_scope = 1:numel(data_scope)
                load_begin = data_scope{load_scope}(1);
                load_end = data_scope{load_scope}(2);
                data_num = load_end-load_begin+1;
                total_data_num = total_data_num + data_num;
                for data_loop = 1:numel(data)
                    train_time = train_time+1;
                    load_bias_amp
                    %% Initialize network
                    miniBatchSize = floor(numel(xTrain)/40);
                    validationFrequency = floor(numel(xTrain)/miniBatchSize/2);

                    layers = [...
                        sequenceInputLayer(inputSize)
                        fullyConnectedLayer(numHiddenUnits)
                        fullyConnectedLayer(numHiddenUnits)
                        reluLayer % 1
                        fullyConnectedLayer(numHiddenUnits)
                        reluLayer % 2
                        fullyConnectedLayer(numHiddenUnits)
                        sigmoidLayer % 3
                        %     fullyConnectedLayer(numHiddenUnits)
                        %     reluLayer % 4
                        %     fullyConnectedLayer(numHiddenUnits)
                        %     reluLayer % 5
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
                    for i = 1:test_num
                        eval([['nmse_valid',num2str(i),'_mat'],'= zeros(1,1);']);
                    end
                    xTrain_reshape = reshape(xTrain,[],test_num);
                    yTrain_reshape = reshape(yTrain,[],test_num);
                    for i = 1:test_num
                        xValid_i = ['xValid',num2str(i)];
                        eval([xValid_i ,'= xTrain_reshape(:,i);']);
                        yValid_i = ['yValid',num2str(i)];
                        eval([yValid_i ,'= yTrain_reshape(:,i);']);
                    end
                    if train_time<=1
                        net_path = savePath_mat+"/net/net"+train_time;
                        if(~exist(net_path,'dir'))
                            mkdir(char(net_path));
                        end

                        net = trainNetwork(xTrain,yTrain,layers,options);
                        save(net_path+"/net.mat",'net');  % Save the trained network

                        for j = 1:test_num
                            x_valid = eval(['xValid',num2str(j)]).';
                            y_valid = eval(['yValid',num2str(j)]).';

                            y_hat_valid = predict(net,x_valid,'MiniBatchSize',miniBatchSize);
                            y_hatT_valid = y_hat_valid.';

                            nmseNum_valid = cellfun(@(hat,exp)10*log10(sum(sum((hat-exp).^2))/sum(sum(exp.^2))),y_hatT_valid ,y_valid);
                            nmse_loop_valid = mean(nmseNum_valid);
                            eval([['nmse_valid',num2str(j),'_mat'],'=nmse_loop_valid;']);
                        end
                        fprintf("already training %d times \n",train_time);

                        nmse_valid_mean = zeros(1,test_num);
                        for i = 1:test_num
                            nmse_valid_mean_tem = mean(eval(['nmse_valid',num2str(i),'_mat']));
                            nmse_valid_mean(i) = nmse_valid_mean_tem;
                        end

                        for i = 1:test_num
                            if i == 1
                                save_Nmse_valid = fopen(net_path+"/save_Nmse_valid.txt",'w');
                                save_amp_bias_txt = fopen(net_path+"/save_amp.txt",'w');
                                %         save(savePath_mat+"/save_Nmse.mat",save_nmse_name);
                            else
                                save_Nmse_valid = fopen(net_path+"/save_Nmse_valid.txt",'a');
                                save_amp_bias_txt = fopen(net_path+"/save_amp.txt",'a');
                                %         save(savePath_mat+"/save_Nmse.mat",save_nmse_name,'-append');
                            end
                            fprintf(save_Nmse_valid,"%f \n" , nmse_valid_mean(i));
                            fprintf(save_amp_bias_txt," amp = %f , bias = %f ,bandpower = %f \n" , save_amp(i), bias_save(i), band_power(i));
                            if i == test_num
                                fprintf(save_amp_bias_txt," load begin = %d , load end = %d  \n" , load_begin,load_end);
                            end
                            fclose(save_Nmse_valid);
                            fclose(save_amp_bias_txt);
                        end

                        for i =1:test_num
                            eval(['clear xTest',num2str(i)])
                            eval(['clear yTest',num2str(i)])
                            eval(['clear xValid',num2str(i)])
                            eval(['clear yValid',num2str(i)])
                        end
                        clear xTrain yTrain
                        pause(10)
                    else
                        net_path = savePath_mat+"/net/net"+train_time;
                        if(~exist(net_path,'dir'))
                            mkdir(char(net_path));
                        end

                        load(savePath_mat+"/net/net"+(train_time-1)+"/net.mat");
                        layers = net.Layers;
                        net = trainNetwork(xTrain,yTrain,layers,options);
                        save(net_path+"/net.mat",'net');  % Save the trained network

                        for j = 1:test_num
                            x_valid = eval(['xValid',num2str(j)]).';
                            y_valid = eval(['yValid',num2str(j)]).';

                            y_hat_valid = predict(net,x_valid,'MiniBatchSize',miniBatchSize);
                            y_hatT_valid = y_hat_valid.';

                            nmseNum_valid = cellfun(@(hat,exp)10*log10(sum(sum((hat-exp).^2))/sum(sum(exp.^2))),y_hatT_valid ,y_valid);
                            nmse_loop_valid = mean(nmseNum_valid);
                            eval([['nmse_valid',num2str(j),'_mat(',num2str(i),')'],'=nmse_loop_valid;']);
                        end
                        fprintf("already training %d times \n",train_time);

                        nmse_valid_mean = zeros(1,test_num);
                        for i = 1:test_num
                            nmse_valid_mean_tem = mean(eval(['nmse_valid',num2str(i),'_mat']));
                            nmse_valid_mean(i) = nmse_valid_mean_tem;
                        end

                        for i = 1:test_num
                            if i == 1
                                save_Nmse_valid = fopen(net_path+"/save_Nmse_valid.txt",'w');
                                save_amp_bias_txt = fopen(net_path+"/save_amp.txt",'w');
                                %         save(savePath_mat+"/save_Nmse.mat",save_nmse_name);
                            else
                                save_Nmse_valid = fopen(net_path+"/save_Nmse_valid.txt",'a');
                                save_amp_bias_txt = fopen(net_path+"/save_amp.txt",'a');
                                %         save(savePath_mat+"/save_Nmse.mat",save_nmse_name,'-append');
                            end
                            fprintf(save_Nmse_valid,"%f \n" , nmse_valid_mean(i));
                            fprintf(save_amp_bias_txt," amp = %f , bias = %f ,bandpower = %f \n" , save_amp(i), bias_save(i), band_power(i));
                            if i == test_num
                                fprintf(save_amp_bias_txt," load begin = %d , load end = %d  \n" , load_begin,load_end);
                            end
                            fclose(save_Nmse_valid);
                            fclose(save_amp_bias_txt);
                        end

                        for i =1:test_num
                            eval(['clear xTest',num2str(i)])
                            eval(['clear yTest',num2str(i)])
                            eval(['clear xValid',num2str(i)])
                            eval(['clear yValid',num2str(i)])
                        end
                        clear xTrain yTrain
                        pause(10)
                    end
                end
            end
        end
    end
    
end

%% Save data

save_parameter = fopen(savePath_txt+"/save_parameter.txt",'w');
fprintf(save_parameter,"\n \n");
fprintf(save_parameter," Default training \n");
fprintf(save_parameter," Threenonlinear ,\r\n ini learningRate = %e ,\r\n min batch size = %d , \r\n " ...
    ,inilearningRate, miniBatchSize);
fprintf(save_parameter,"DropPeriod = %d , DropFactor = %f ,\r\n ",LearnRateDropPeriod, LearnRateDropFactor);
fprintf(save_parameter,"amp =");
for i = 1:length(amp_scope_ini)
    fprintf(save_parameter," %f,",amp_scope_ini(i));
end
fprintf(save_parameter,"\r\n");
fprintf(save_parameter," bias =");
for i = 1:length(bias_scope)
    fprintf(save_parameter," %f,",bias_scope(i));
end
fprintf(save_parameter,"\r\n");
fprintf(save_parameter," data num = %d , split num = %d , train num = %d\r\n",total_data_num,split_num,total_data_num*split_num*train_percent);
fprintf(save_parameter," validationFrequency is floor(numel(xTrain)/miniBatchSize/2) \n");
fprintf(save_parameter," origin rate = %e , receive rate = %e \n",ori_rate,rec_rate);
fprintf(save_parameter," H order = %d ,related num = %d \n",h_order,related_num);
fprintf(save_parameter," Hidden Units = %d \n",numHiddenUnits);
fprintf(save_parameter," Add zero num = %d \n",add_zero);
fclose(save_parameter);


fprintf("\n Training end ..." + ...
    "\n Threenonlinear , ini learningRate = %e , min batch size = %d , DropPeriod = %d , DropFactor = %f , data_num = %d \n",...
    inilearningRate, miniBatchSize, LearnRateDropPeriod, LearnRateDropFactor, data_num);
fprintf(" result saved in %s \n",savePath_mat);


    
