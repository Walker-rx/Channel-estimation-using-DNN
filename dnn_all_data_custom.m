clear
close all

tStart = tic;

t = datetime('now');
folder = '4.14';
save_path = "data_save/light_data_"+folder;

ver = 14;

savePath_txt = save_path + "/result1/"+t.Month+"."+t.Day+"/mix_bias_amp/Threenonlinear"+ver;   
savePath_mat = save_path + "/result1/"+t.Month+"."+t.Day+"/mix_bias_amp/Threenonlinear"+ver; 
if(~exist(savePath_txt,'dir'))
    mkdir(char(savePath_txt));
end
if(~exist(savePath_mat,'dir'))
    mkdir(char(savePath_mat));
end

%% Network parameters
bias_scope = 0.05:0.04:0.85;
amp_scope_ini = [0.1613 0.32106 0.48082 0.64058 0.8003 1];
% bias_scope = 0.45;
% amp_scope_ini = 1;

total_cell = 60;
total_data_num = total_cell;
loop_data_num = 30;
if loop_data_num>30
    loop_data_num = 30;
end
loop_train_num = ceil(total_cell/loop_data_num);

data_scope = cell(1,loop_train_num);
for i = 1:loop_train_num
    if i == loop_train_num
        data_scope{i} = [(i-1)*loop_data_num+1 , total_cell];
    else
        data_scope{i} = [(i-1)*loop_data_num+1 , i*loop_data_num];
    end   
end

total_loop_time = 1;
train_percent = 0.95;

for train_loop_time = 1:total_loop_time
    train_time = 0;
    total_data_num = 0;
    total_loss = {};
    total_learnRate = {};

    amp_data = [];
%     bias_data = cell(1,2);
    amp_scope = amp_scope_ini;
    while ~isempty(amp_scope)
        if length(amp_scope)>1
            amp_order = randperm(length(amp_scope),2);
            amp_data(1) = amp_scope(amp_order(1));
            amp_data(2) = amp_scope(amp_order(2));
            amp_scope(amp_order) = [];

            bias_scope_tmp = bias_scope;
            while ~isempty(bias_scope_tmp)
                [data,bias_scope_tmp] = split_data_custom(amp_data,bias_scope_tmp);         

                for load_scope = 1:numel(data_scope)
                    load_begin = data_scope{load_scope}(1);
                    load_end = data_scope{load_scope}(2);
                    data_num = load_end-load_begin+1;
                    
                    for data_loop = 1:numel(data)
                        clearvars -except total_loop_time train_loop_time load_scope save_path savePath_mat savePath_txt ...
                            bias_scope amp_scope_ini data_scope loop_train_num train_percent train_time total_data_num total_loss total_learnRate...
                            data data_loop amp_scope bias_scope_tmp velocity data_num load_begin load_end averageGrad averageSqGrad tStart tic
                        pause(10)
                        ori_rate = 10e6;
                        rec_rate = 60e6;
                        rate_times = rec_rate/ori_rate;
                        related_num = 8;
                        h_order = rate_times*related_num;
                        add_zero = h_order/2;

                        split_num = 10;  % Cut a signal into split_num shares

                        inputSize = h_order+1;
%                         numHiddenUnits = 60;
                        numHiddenUnits = 200;
                        outputSize = rate_times;  % y=h*x+n;  y:(outputSize,m) h:(outputSize,inputSize) x:(inputSize,m)
                        maxEpochs = 60;
                        LearnRateDropPeriod = 8;
                        LearnRateDropFactor = 0.5;
                        inilearningRate = 1e-2;
                        velocity = [];
                        momentum = 0.9;
                        train_time = train_time+1;

                        load_bias_amp_custom
                        
                        for clear_loop = 1:test_num
                            eval("clear xTest"+clear_loop);
                            eval("clear yTest"+clear_loop);
                        end
                        clear xTrain_tmp yTrain_tmp xtop_tem ytop_tem

                        %% Initialize network
                        xTrain = cell2mat(xTrain);
                        yTrain = cell2mat(yTrain);
                        xValidation = cell2mat(xValidation);
                        yValidation = cell2mat(yValidation);
                        shuffle_index_valid = randperm(size(xValidation,2));
                        xValidation = xValidation(:,shuffle_index_valid);
                        yValidation = yValidation(:,shuffle_index_valid);

                        numOber = size(xTrain,2);
                        %         miniBatchSize = 80000/4;
                        %         numIterPerEpoch = floor(numOber/miniBatchSize);
                        %         validationFrequency = floor(numel(xTrain)/miniBatchSize/2);
                        numIterPerEpoch = 400;
                        miniBatchSize = floor(numOber/numIterPerEpoch);
                        validationFrequency = floor(numIterPerEpoch/4);

                        layers = [...
                            sequenceInputLayer(inputSize)
                            fullyConnectedLayer(numHiddenUnits)
                            fullyConnectedLayer(numHiddenUnits)
                            reluLayer % 1
                            fullyConnectedLayer(numHiddenUnits)
                            reluLayer % 2
                            fullyConnectedLayer(numHiddenUnits)
                            sigmoidLayer % 3
                            fullyConnectedLayer(outputSize)];

                        lgraph = layerGraph(layers);
                        dlnet = dlnetwork(lgraph);

                        %% Train network                  
                        net_path = savePath_mat+"/net/looptime"+train_loop_time+"/net"+train_time;
                        if(~exist(net_path,'dir'))
                            mkdir(char(net_path));
                        end
                        
                        tic
                        [ dlnet, velocity, losss, learnRate_save ] = dnn_train_custom(maxEpochs, numOber, xTrain, yTrain, ...
                                                        xValidation, yValidation, numIterPerEpoch, miniBatchSize, dlnet, ...
                            velocity, inilearningRate, momentum,train_loop_time, train_time, LearnRateDropPeriod, LearnRateDropFactor, validationFrequency);
                        toc
                        total_loss{train_time} = losss.';
                        total_learnRate{train_time} = learnRate_save.';
                        save(net_path+"/net.mat",'dlnet');  % Save the trained network
                    end           % for data_loop = 1:numel(data) ; amp_scope size > 1
                end               % for load_scope = 1:numel(data_scope) ; amp_scope size > 1
            end                   % while ~isempty(bias_scope_tmp) ; amp_scope size > 1
            
        else

            amp_data(1) = amp_scope(1);
            amp_scope(1) = [];
            bias_scope_tmp = bias_scope;
            while ~isempty(bias_scope_tmp)
                [data,bias_scope_tmp] = split_data_custom(amp_data,bias_scope_tmp);         

                for load_scope = 1:numel(data_scope)
                    load_begin = data_scope{load_scope}(1);
                    load_end = data_scope{load_scope}(2);
                    data_num = load_end-load_begin+1;
                    total_data_num = total_data_num + data_num;
                    for data_loop = 1:numel(data)
                        clearvars -except total_loop_time train_loop_time load_scope save_path savePath_mat savePath_txt ...
                            bias_scope amp_scope_ini data_scope loop_train_num train_percent train_time total_data_num total_loss total_learnRate...
                            data data_loop amp_scope bias_scope_tmp velocity data_num load_begin load_end averageGrad averageSqGrad tStart tic
                        pause(10)
                        ori_rate = 10e6;
                        rec_rate = 60e6;
                        rate_times = rec_rate/ori_rate;
                        related_num = 8;
                        h_order = rate_times*related_num;
                        add_zero = h_order/2;

                        split_num = 10;  % Cut a signal into split_num shares

                        inputSize = h_order+1;
%                         numHiddenUnits = 60;
                        numHiddenUnits = 200;
                        outputSize = rate_times;  % y=h*x+n;  y:(outputSize,m) h:(outputSize,inputSize) x:(inputSize,m)
                        maxEpochs = 60;
                        LearnRateDropPeriod = 8;
                        LearnRateDropFactor = 0.5;
                        inilearningRate = 1e-2;
                        velocity = [];
                        momentum = 0.9;
                        train_time = train_time+1;
                        load_bias_amp_custom
                        for clear_loop = 1:test_num
                            eval("clear xTest"+clear_loop);
                            eval("clear yTest"+clear_loop);
                        end
                        clear xTrain_tmp yTrain_tmp xtop_tem ytop_tem

                        %% Initialize network
                        xTrain = cell2mat(xTrain);
                        yTrain = cell2mat(yTrain);
                        numOber = size(xTrain,2);
                        %         miniBatchSize = 80000/4;
                        %         numIterPerEpoch = floor(numOber/miniBatchSize);
                        %         validationFrequency = floor(numel(xTrain)/miniBatchSize/2);
                        numIterPerEpoch = 400;
                        miniBatchSize = floor(numOber/numIterPerEpoch);
                        validationFrequency = floor(numIterPerEpoch/4);

                        layers = [...
                            sequenceInputLayer(inputSize)
                            fullyConnectedLayer(numHiddenUnits)
                            fullyConnectedLayer(numHiddenUnits)
                            reluLayer % 1
                            fullyConnectedLayer(numHiddenUnits)
                            reluLayer % 2
                            fullyConnectedLayer(numHiddenUnits)
                            sigmoidLayer % 3
                            fullyConnectedLayer(outputSize)];

                        lgraph = layerGraph(layers);
                        dlnet = dlnetwork(lgraph);

                        %% Train network                  
                        net_path = savePath_mat+"/net/looptime"+train_loop_time+"/net"+train_time;
                        if(~exist(net_path,'dir'))
                            mkdir(char(net_path));
                        end

                        tic
                        [ dlnet, velocity, losss, learnRate_save ] = dnn_train_custom(maxEpochs, numOber, xTrain, yTrain, ...
                                                        xValidation, yValidation, numIterPerEpoch, miniBatchSize, dlnet, ...
                            velocity, inilearningRate, momentum,train_loop_time, train_time, LearnRateDropPeriod, LearnRateDropFactor, validationFrequency);
                        toc
                        total_loss{train_time} = losss.';
                        total_learnRate{train_time} = learnRate_save.';
                        save(net_path+"/net.mat",'dlnet');  % Save the trained network

                    end           % for data_loop = 1:numel(data) ; amp_scope size = 1
                end               % for load_scope = 1:numel(data_scope) ; amp_scope size = 1
            end                   % while ~isempty(bias_scope_tmp) ; amp_scope size = 1     

        end              % if length(amp_scope)>1
        
    end      % while ~isempty(amp_scope)

end

                   
%% Save data
save_parameter = fopen(savePath_txt+"/save_parameter.txt",'w');
fprintf(save_parameter,"\n \n");
fprintf(save_parameter," Custom training \n");
fprintf(save_parameter," Threenonlinear ,\r\n 400 iteration per epoch , \r\n ");
fprintf(save_parameter,"ini learningRate = %e ,\r\n DropPeriod = %d , DropFactor = %f ,\r\n ",inilearningRate,LearnRateDropPeriod, LearnRateDropFactor);
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
fprintf(save_parameter," data num = %d , split num = %d , train num = %d\r\n",total_cell,split_num,total_cell*split_num*train_percent);
fprintf(save_parameter," validationFrequency is floor(numIterPerEpoch/4) \n");
fprintf(save_parameter," origin rate = %e , receive rate = %e \n",ori_rate,rec_rate);
fprintf(save_parameter," H order = %d ,related num = %d \n",h_order,related_num);
fprintf(save_parameter," Hidden Units = %d \n",numHiddenUnits);
fprintf(save_parameter," Add zero num = %d \n",add_zero);
fclose(save_parameter);


fprintf("\n Training end ..." + ...
    "\n Threenonlinear , ini learningRate = %e ,  DropPeriod = %d , DropFactor = %f , data_num = %d \n",...
    inilearningRate,  LearnRateDropPeriod, LearnRateDropFactor, total_data_num);
fprintf(" result saved in %s \n",savePath_mat);

tEnd = toc(tStart);
disp("Total using "+floor(tEnd/60)+"min "+mod(tEnd,60)+"s")


    
