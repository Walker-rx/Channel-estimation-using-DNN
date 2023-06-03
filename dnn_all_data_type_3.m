clear
close all

tStart = tic;

t = datetime('now');
folder = '5.21';
load_path_ini = "/home/xliangseu/ruoxu/equalization-using-DNN/data_save/light_data_"+folder;
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
bias_scope = 0.05:0.04:0.85;
amp_scope_ini = [0.005 0.007 0.015 0.024 0.034 0.045 0.08 0.18 0.25 0.3];
% amp_scope_ini = [0.1613 0.32106 0.48082 0.64058 0.8003 1];

total_cell = 60;
total_data_num = total_cell;

loop_data_num = ceil(21*50/(length(bias_scope)*length(amp_scope_ini)))+1;
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

    for load_scope = 1:numel(data_scope)
        clearvars -except total_loop_time train_loop_time load_scope save_path savePath_mat savePath_txt ...
            bias_scope amp_scope_ini data_scope loop_train_num train_percent train_time total_data_num total_loss total_learnRate ...
            velocity averageGrad averageSqGrad tStart tic load_path_ini
        pause(10)
        %%
        ori_rate = 10e6;
        rec_rate = 60e6;
        rate_times = rec_rate/ori_rate;
        related_num = 8;
        h_order = rate_times*related_num;
        add_zero = h_order/2;

        split_num = 10;  % Cut a signal into split_num shares

        inputSize = h_order+1;
%         numHiddenUnits = 60;
        numHiddenUnits = 200;
        outputSize = rate_times;  % y=h*x+n;  y:(outputSize,m) h:(outputSize,inputSize) x:(inputSize,m)
        maxEpochs = 60;
        LearnRateDropPeriod = 8;
        LearnRateDropFactor = 0.5;
        inilearningRate = 1e-2;
        velocity = [];
        momentum = 0.9;

        %% Load data
        data = split_data_custom_2(amp_scope_ini,bias_scope);       
        load_begin = data_scope{load_scope}(1);
        load_end = data_scope{load_scope}(2);
        data_num = load_end-load_begin+1;

        train_time = train_time+1;
        xValidation = {};
        yValidation = {};

        xTrain = {};
        yTrain = {};
        for data_loop = 1:numel(data)
            load_bias_amp_custom_2
        end

        %% Shuffling data
        Train_cell_num = numel(xTrain);
        X = xTrain;
        Y = yTrain;
        X = cell2mat(X);
        Y = cell2mat(Y);
        numOber = size(X,2);
        idx = randperm(numOber);
        X = X(:,idx);
        Y = Y(:,idx);
        X = mat2cell(X,size(X,1),repmat(size(X,2)/Train_cell_num,1,Train_cell_num));
        Y = mat2cell(Y,size(Y,1),repmat(size(Y,2)/Train_cell_num,1,Train_cell_num));
        xTrain = X;
        yTrain = Y;

        Validation_cell_num = numel(xValidation);
        X = xValidation;
        Y = yValidation;
        X = cell2mat(X);
        Y = cell2mat(Y);
        numOber = size(X,2);
        idx = randperm(numOber);
        X = X(:,idx);
        Y = Y(:,idx);
        X = mat2cell(X,size(X,1),repmat(size(X,2)/Validation_cell_num,1,Validation_cell_num));
        Y = mat2cell(Y,size(Y,1),repmat(size(Y,2)/Validation_cell_num,1,Validation_cell_num));
        xValidation = X;
        yValidation = Y;
        clear X Y idx

        %% Initialize network
        miniBatchSize = floor(numel(xTrain)/400);
        validationFrequency = floor(numel(xTrain)/miniBatchSize/4);
        dnn_option

        %% Train network
        dnn_train_default(train_time, savePath_mat, xTrain, yTrain, layers, options, test_num,...
            save_amp, bias_save, band_power, load_begin, load_end);
            
    end
end
%% Save data

save_parameter = fopen(savePath_txt+"/save_parameter.txt",'w');
fprintf(save_parameter,"\n \n");
fprintf(save_parameter," Type_3 training \n");
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
    "\n Threenonlinear , ini learningRate = %e , min batch size = %d , DropPeriod = %d , DropFactor = %f , data_num = %d \n",...
    inilearningRate, miniBatchSize, LearnRateDropPeriod, LearnRateDropFactor, data_num);
fprintf(" result saved in %s \n",savePath_mat);

tEnd = toc(tStart);
disp("Total using "+floor(tEnd/60)+"min "+mod(tEnd,60)+"s")


    
