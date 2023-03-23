clear
close all

t = datetime('now');
folder = '3.22';
save_path = "data_save/light_data_"+folder;

%% Network parameters
ori_rate = 10e6;
rec_rate = 60e6;
rate_times = rec_rate/ori_rate;
related_num = 8;
h_order = rate_times*related_num;
add_zero = h_order/2;

data_num = 100;
split_num = 10;  % Cut a signal into split_num shares

inputSize = h_order+1;
numHiddenUnits = 60;
outputSize = rate_times;  % y=h*x+n;  y:(outputSize,m) h:(outputSize,inputSize) x:(inputSize,m)
maxEpochs = 2000;
miniBatchSize = 400;
LearnRateDropPeriod = 8;
LearnRateDropFactor = 0.1;
inilearningRate = 1e-2;

ver = 1;

%% Loop parameter settings
data_type = 1;
if data_type == 1
    bias_begin = 0.1;
    bias_step = 0.05;
    bias_end = 0.8;
    bias_loop_num = (bias_end-bias_begin)/bias_step+1;

    amp_loop_begin = 1;
    amp_loop_end = 1;
    amp_loop_step = 1;
    amp_loop_num = (amp_loop_end - amp_loop_begin)/amp_loop_step + 1 ;


    amp_begin = 1;
    amp_norm = 0;
elseif data_type == 2
    bias_begin = 0.05;
    bias_step = 0.04;
    bias_end = 0.85;
    bias_loop_num = (bias_end-bias_begin)/bias_step+1;

    amp_loop_begin = 1;
    amp_loop_end = 1;
    amp_loop_step = 1;
    amp_loop_num = (amp_loop_end - amp_loop_begin)/amp_loop_step + 1 ;


    amp_begin = 0.1613;
    amp_norm = 0;
end

%%
fprintf("This is Threenonlinear network , ini learningRate = %e , min batch size = %d , DropPeriod = %d , DropFactor = %f \n",...
    inilearningRate,miniBatchSize,LearnRateDropPeriod,LearnRateDropFactor);
fprintf("Hidden Units = %d , v%d \n",numHiddenUnits,ver)

%% Load data
test_num = 0;
bias_loop = 0;
save_amp = zeros(1,round(bias_loop_num*amp_loop_num));
band_power = zeros(1,round(bias_loop_num*amp_loop_num));
for bias = bias_begin : bias_step :bias_end
    
    bias_loop = bias_loop + 1;
    test_num_amp = 0;

    for loop = amp_loop_begin : amp_loop_step :amp_loop_end
        test_num_amp = test_num_amp + 1;
        load_path = save_path + "/data"+data_type+"/10M/bias"+bias+"/amp"+loop+"/mat";
        fprintf(" bias = %f , load amp = %d \n",bias,loop);
        load_data
        totalNum = data_num*split_num;
        trainNum = floor(totalNum*0.95);
        xTrain_tmp = x(1:trainNum);
        yTrain_tmp = y(1:trainNum);
        xTest_amp_tmp = x(trainNum+1:end);
        yTest_amp_tmp = y(trainNum+1:end);

        %     xTrain_tmp = cellfun(@(cell1)(cell1*100*1.1^amp),xTrain_tmp,'UniformOutput',false);
        %     xTest_tmp = cellfun(@(cell1)(cell1*100*1.1^amp),xTest_tmp,'UniformOutput',false);

        amp_loop = 32000*(amp_begin+(loop-1)*amp_norm);
        save_amp((bias_loop-1)*amp_loop_num+test_num_amp) = 10*log10(amp_loop^2);
        xTrain_tmp = cellfun(@(cell1)(cell1*amp_loop),xTrain_tmp,'UniformOutput',false);
        xTest_amp_tmp = cellfun(@(cell1)(cell1*amp_loop),xTest_amp_tmp,'UniformOutput',false);

        xTest_amp_name = ['xTest_amp_',num2str(test_num_amp)];
        yTest_amp_name = ['yTest_amp_',num2str(test_num_amp)];
        eval([xTest_amp_name,'=xTest_amp_tmp;']);
        eval([yTest_amp_name,'=yTest_amp_tmp;']);

        if loop == amp_loop_begin
            xTrain_amp = xTrain_tmp;
            yTrain_amp = yTrain_tmp;
        else
            xTrain_amp = [xTrain_amp xTrain_tmp];
            yTrain_amp = [yTrain_amp yTrain_tmp];
        end
        clear x y
    end

    %%  Normalize data
    totaltrain = numel(xTrain_amp);
    % norm_cell = xTrain{floor(totaltrain/2)};
    % norm_factor = 1/norm(norm_cell)*sqrt(length(norm_cell));
    % load_path = "data_save/light_data_2.28/result/3.1/25M/8pam/mix_amp/Twononlinear";

    load_path = "data_save/light_data_3.10/data/10M/rand_bias0.3/";
    norm_mat = load(load_path+"/save_norm.mat");
    norm_names = fieldnames(norm_mat);
    norm_factor = gather(eval(strcat('norm_mat.',norm_names{1})));

    xTrain_amp = cellfun(@(cell1)(cell1*norm_factor),xTrain_amp,'UniformOutput',false);

    for i = 1:amp_loop_num
        band_power((bias_loop-1)*amp_loop_num+i) = bandpower(xTrain_amp{10+(i-1)*trainNum});
    end

    for i = 1:test_num_amp
        xTest_nor = eval(['xTest_amp_',num2str(i)]);
        xTest_nor = cellfun(@(cell1)(cell1*norm_factor),xTest_nor,'UniformOutput',false);
        eval([['xTest_amp_',num2str(i)],'= xTest_nor;']);
    end

    %%  Reshape data
    for i = 1:numel(xTrain_amp)
        xTrain_amp{i} = toeplitz(xTrain_amp{i}(h_order:-1:1),xTrain_amp{i}(h_order:end));
        xTrain_amp{i} = [xTrain_amp{i}; bias*ones(1,size(xTrain_amp{i},2) )];
        yTrain_amp{i} = reshape(yTrain_amp{i}(1:split_length*rate_times),outputSize,split_length);
        yTrain_amp{i} = yTrain_amp{i}(:,1:size(xTrain_amp{i},2));
    end
    for i = 1:test_num_amp
        xtop_tem = eval(['xTest_amp_',num2str(i)]);
        ytop_tem = eval(['yTest_amp_',num2str(i)]);
        for j = 1:numel(xtop_tem)
            xtop_tem{j} = toeplitz(xtop_tem{j}(h_order:-1:1),xtop_tem{j}(h_order:end));
            xtop_tem{j} = [xtop_tem{j}; bias*ones(1,size(xtop_tem{j},2) )];
            ytop_tem{j} = reshape(ytop_tem{j}(1:split_length*rate_times),outputSize,split_length);
            ytop_tem{j} = ytop_tem{j}(:,1:size(xtop_tem{j},2));
            %             xTest{i} = toeplitz(xTest{i}(inputSize:-1:1),xTest{i}(inputSize:end));
            %             yTest{i} = reshape(yTest{i}(1:6000),outputSize,1000);
            %             yTest{i} = yTest{i}(:,1:size(xTest1{i},2));
        end
        eval([['xTest_amp_',num2str(i)],'= xtop_tem;']);
        eval([['yTest_amp_',num2str(i)],'= ytop_tem;']);
    end

    %%
    test_num = test_num + test_num_amp;
    if bias == bias_begin
        xTrain = xTrain_amp;
        yTrain = yTrain_amp;
    else
        xTrain = [xTrain  xTrain_amp];
        yTrain = [yTrain  yTrain_amp];
    end

    for i = 1:test_num_amp
        xTest_name = ['xTest',num2str((bias_loop-1)*test_num_amp+i)];
        yTest_name = ['yTest',num2str((bias_loop-1)*test_num_amp+i)];
        xTest_bias_tmp = eval(['xTest_amp_',num2str(i)]);
        yTest_bias_tmp = eval(['yTest_amp_',num2str(i)]);
        eval([xTest_name,'=xTest_bias_tmp;']);
        eval([yTest_name,'=yTest_bias_tmp ;']);
    end
    
end

%%
dnn_all_bias2

%%
test_num = test_num + test_num2;
save_amp = [save_amp save_amp2];
band_power = [band_power band_power2];

%% Initialize network
% validationFrequency = floor(size(xTrain{1},2)/miniBatchSize);
% validationFrequency = floor(size(xTrain{1},2)/100);
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
looptime = 2;
for i = 1:test_num
    eval([['nmse',num2str(i),'_mat'],'= zeros(1,looptime);']);
    eval([['nmse_valid',num2str(i),'_mat'],'= zeros(1,looptime);']);
end

xTrain_reshape = reshape(xTrain,[],test_num);
yTrain_reshape = reshape(yTrain,[],test_num);
for i = 1:test_num
    xValid_i = ['xValid',num2str(i)];
    eval([xValid_i ,'= xTrain_reshape(:,i);']);
    yValid_i = ['yValid',num2str(i)];
    eval([yValid_i ,'= yTrain_reshape(:,i);']);
end

for i = 1:looptime    
    net = trainNetwork(xTrain,yTrain,layers,options);
    h = net.Layers(2).Weights.';
    for j = 1:test_num
        x_fortest = eval(['xTest',num2str(j)]);
        y_fortest = eval(['yTest',num2str(j)]);
        x_valid = eval(['xValid',num2str(j)]).';
        y_valid = eval(['yValid',num2str(j)]).';

        y_hat = predict(net,x_fortest,'MiniBatchSize',miniBatchSize);
        y_hatT = y_hat.';
        y_hat_valid = predict(net,x_valid,'MiniBatchSize',miniBatchSize);
        y_hatT_valid = y_hat_valid.';

        nmseNum = cellfun(@(hat,exp)10*log10(sum(sum((hat-exp).^2))/sum(sum(exp.^2))),y_hatT,y_fortest);
        nmse_loop = mean(nmseNum);
        eval([['nmse',num2str(j),'_mat(',num2str(i),')'],'=nmse_loop;']);
        nmseNum_valid = cellfun(@(hat,exp)10*log10(sum(sum((hat-exp).^2))/sum(sum(exp.^2))),y_hatT_valid ,y_valid);
        nmse_loop_valid = mean(nmseNum_valid);
        eval([['nmse_valid',num2str(j),'_mat(',num2str(i),')'],'=nmse_loop_valid;']);
    end       
    fprintf("already training %d times \n",i);
end
nmse_mean = zeros(1,test_num);
nmse_valid_mean = zeros(1,test_num);
for i = 1:test_num
    nmse_mean_tem = mean(eval(['nmse',num2str(i),'_mat']));
    nmse_mean(i) = nmse_mean_tem;
    nmse_valid_mean_tem = mean(eval(['nmse_valid',num2str(i),'_mat']));
    nmse_valid_mean(i) = nmse_valid_mean_tem;
end

%% Save data
savePath_txt = save_path + "/result3/"+t.Month+"."+t.Day+"/10M/mix_bias/mix_amp/Threenonlinear"+ver;   
savePath_mat = save_path + "/result3/"+t.Month+"."+t.Day+"/10M/mix_bias/mix_amp/Threenonlinear"+ver; 

if(~exist(savePath_txt,'dir'))
    mkdir(char(savePath_txt));
end
if(~exist(savePath_mat,'dir'))
    mkdir(char(savePath_mat));
end
save_parameter = fopen(savePath_txt+"/save_parameter.txt",'w');
fprintf(save_parameter,"\n \n");
fprintf(save_parameter," Threenonlinear ,\r\n ini learningRate = %e ,\r\n min batch size = %d , \r\n " ...
    ,inilearningRate, miniBatchSize);
fprintf(save_parameter,"DropPeriod = %d , DropFactor = %f ,\r\n ",LearnRateDropPeriod, LearnRateDropFactor);
fprintf(save_parameter,"amp begin = %d , amp end = %d , amp step = %d \r\n ",amp_loop_begin, amp_loop_end, amp_loop_step);
fprintf(save_parameter,"data_num = %d , split num = %d , train num = %d\r\n",data_num,split_num,trainNum*2);
fprintf(save_parameter," validationFrequency is floor(numel(xTrain)/miniBatchSize/4) \n");
fprintf(save_parameter," origin rate = %e , receive rate = %e \n",ori_rate,rec_rate);
fprintf(save_parameter," H order = %d ,related num = %d \n",h_order,related_num);
fprintf(save_parameter," Hidden Units = %d \n",numHiddenUnits);
fprintf(save_parameter," Add zero num = %d \n",add_zero);
fclose(save_parameter);

save_nmse_name = 'save_nmse';
eval([save_nmse_name,'=nmse_mean;']);
save(savePath_mat+"/save_Nmse.mat",save_nmse_name);
save_nmse_valid_name = 'save_nmse_valid';
eval([save_nmse_valid_name,'=nmse_valid_mean;']);
save(savePath_mat+"/save_Nmse_valid.mat",save_nmse_valid_name);
 
save(savePath_mat+"/net.mat",'net');  % Save the trained network

for i = 1:test_num
    if i == 1
        save_Nmse = fopen(savePath_txt+"/save_Nmse.txt",'w');
        save_Nmse_valid = fopen(savePath_txt+"/save_Nmse_valid.txt",'w');
        save_bandpower = fopen(savePath_txt+"/save_bandpower.txt",'w');
        save_amp_txt = fopen(savePath_txt+"/save_amp.txt",'w');
%         save(savePath_mat+"/save_Nmse.mat",save_nmse_name);
    else
        save_Nmse = fopen(savePath_txt+"/save_Nmse.txt",'a');
        save_Nmse_valid = fopen(savePath_txt+"/save_Nmse_valid.txt",'a');
        save_bandpower = fopen(savePath_txt+"/save_bandpower.txt",'a');
        save_amp_txt = fopen(savePath_txt+"/save_amp.txt",'a');
%         save(savePath_mat+"/save_Nmse.mat",save_nmse_name,'-append');
    end
    fprintf(save_Nmse,"%f \n" , nmse_mean(i));
    fprintf(save_Nmse_valid,"%f \n" , nmse_valid_mean(i));
    fprintf(save_bandpower,"%f \n" , band_power(i));
    fprintf(save_amp_txt,"%f \n" , save_amp(i));
    fclose(save_Nmse);
    fclose(save_Nmse_valid);
    fclose(save_bandpower);
    fclose(save_amp_txt);
end

fprintf(" \n Threenonlinear , ini learningRate = %e , min batch size = %d , DropPeriod = %d , DropFactor = %f , data_num = %d \n",...
    inilearningRate, miniBatchSize, LearnRateDropPeriod, LearnRateDropFactor, data_num);


    
