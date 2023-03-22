clear
close all

t = datetime('now');
save_path = "data_save/light_data_3.16";
% save_path = "data_save/2.23";

%% Network parameters
ori_rate = 10e6;
rec_rate = 60e6;
rate_times = rec_rate/ori_rate;
related_num = 8;
split_num = 10;  % Cut a signal into split_num shares

h_order = rate_times*related_num;
inputSize = h_order;
numHiddenUnits = 60;
outputSize = rate_times;  % y=h*x+n;  y:(outputSize,m) h:(outputSize,inputSize) x:(inputSize,m)
maxEpochs = 2000;
miniBatchSize = 400;
LearnRateDropPeriod = 8;
LearnRateDropFactor = 0.1;
inilearningRate = 1e-2;
ver = 10;
bias = 0.3;
%%
fprintf("This is Threenonlinear network , ini learningRate = %e , min batch size = %d , DropPeriod = %d , DropFactor = %f \n",...
    inilearningRate,miniBatchSize,LearnRateDropPeriod,LearnRateDropFactor);
fprintf("Hidden Units = %d , v%d \n",numHiddenUnits,ver)

%% Load data

loop_step = 2;
test_begin = 1;
test_end = 49;
testLoop_num = (test_end-test_begin)/loop_step+1;
train_begin = 51;
train_end = 51;
trainLoop_num = (train_end-train_begin)/loop_step+1;

amp_begin = 0.0015;
amp_norm = 0.019970; 

save_Testamp = zeros(1,testLoop_num+trainLoop_num);
save_Testamp_log = zeros(1,testLoop_num+trainLoop_num);
save_Trainamp = zeros(1,trainLoop_num);
save_Trainamp_log = zeros(1,trainLoop_num);

test_num = 0;
for loop = test_begin : loop_step :test_end
    test_num = test_num+1;
    load_path = save_path + "/data/10M/rand_bias"+bias+"/amp"+loop+"/mat";   
%     data_num = 20;
    load_data_all
    fprintf(" Test data , load amp=%d , data num = %d \n",loop,data_num);

    xTest_tmp = x;
    yTest_tmp = y;
    
    amp_loop = 32000*(amp_begin+(loop-1)*amp_norm);
    save_Testamp(test_num) = amp_loop;
    save_Testamp_log(test_num) = 10*log10(amp_loop^2);
    xTest_tmp = cellfun(@(cell1)(cell1*amp_loop),xTest_tmp,'UniformOutput',false);
    
    xTest_name = ['xTest',num2str(test_num)];
    yTest_name = ['yTest',num2str(test_num)];
    eval([xTest_name,'=xTest_tmp;']);
    eval([yTest_name,'=yTest_tmp;']);

    clear x y
end
testData_num = data_num;

train_num = 0;
for loop = train_begin : train_end 
    train_num = train_num+1;
    test_num = test_num+1;
    load_path = save_path + "/data/10M/rand_bias"+bias+"/amp"+loop+"/mat";
    load_data_all
    fprintf(" Train data , load amp=%d , data_num = %d \n",loop,data_num);

    totalNum = data_num*split_num;
    trainNum = floor(totalNum*0.95);
    xTrain_tmp = x(1:trainNum);
    yTrain_tmp = y(1:trainNum);
    xTest_tmp = x(trainNum+1:end);
    yTest_tmp = y(trainNum+1:end);
    
    amp_loop = 32000*(amp_begin+(loop-1)*amp_norm);
    save_Trainamp(train_num) = amp_loop;
    save_Testamp(test_num) = amp_loop;
    save_Trainamp_log(train_num) = 10*log10(amp_loop^2);
    save_Testamp_log(test_num) = 10*log10(amp_loop^2);
    xTrain_tmp = cellfun(@(cell1)(cell1*amp_loop),xTrain_tmp,'UniformOutput',false);
    xTest_tmp = cellfun(@(cell1)(cell1*amp_loop),xTest_tmp,'UniformOutput',false);
    
    xTest_name = ['xTest',num2str(test_num)];
    yTest_name = ['yTest',num2str(test_num)];
    eval([xTest_name,'=xTest_tmp;']);
    eval([yTest_name,'=yTest_tmp;']);

    if loop == train_begin
        xTrain = xTrain_tmp;
        yTrain = yTrain_tmp;
    else
        xTrain = [xTrain xTrain_tmp];
        yTrain = [yTrain yTrain_tmp];
    end
    clear x y
end
trainData_num = data_num*0.95;

%%  Normalize data
totaltrain = numel(xTrain);

load_path = "data_save/light_data_3.10/data/10M/rand_bias0.3/";
norm_mat = load(load_path+"/save_norm.mat");
norm_names = fieldnames(norm_mat);
norm_factor = gather(eval(strcat('norm_mat.',norm_names{1})));

xTrain = cellfun(@(cell1)(cell1*norm_factor),xTrain,'UniformOutput',false);

Train_bandpower = zeros(1,trainLoop_num);
for i = 1:trainLoop_num
    Train_bandpower(i) = bandpower(xTrain{10+(i-1)*trainData_num});
end


for i = 1:test_num
    xTest_nor = eval(['xTest',num2str(i)]);
    xTest_nor = cellfun(@(cell1)(cell1*norm_factor),xTest_nor,'UniformOutput',false);
    eval([['xTest',num2str(i)],'= xTest_nor;']);
end

Test_bandpower = zeros(1,test_num);
for i = 1:test_num
    xTest_nor = eval(['xTest',num2str(i)]);
    Test_bandpower(i) = bandpower(xTest_nor{10});
end

%%  Reshape data
for i = 1:numel(xTrain)
    xTrain{i} = toeplitz(xTrain{i}(inputSize:-1:1),xTrain{i}(inputSize:end));
    yTrain{i} = reshape(yTrain{i}(1:split_length*rate_times),outputSize,split_length);
    yTrain{i} = yTrain{i}(:,1:size(xTrain{i},2));
end
for i = 1:test_num
    xtop_tem = eval(['xTest',num2str(i)]);
    ytop_tem = eval(['yTest',num2str(i)]);
    for j = 1:numel(xtop_tem)            
        xtop_tem{j} = toeplitz(xtop_tem{j}(inputSize:-1:1),xtop_tem{j}(inputSize:end));            
        ytop_tem{j} = reshape(ytop_tem{j}(1:split_length*rate_times),outputSize,split_length);
        ytop_tem{j} = ytop_tem{j}(:,1:size(xtop_tem{j},2));
    end
    eval([['xTest',num2str(i)],'= xtop_tem;']);
    eval([['yTest',num2str(i)],'= ytop_tem;']);
end

%% Initialize network

validationFrequency = floor(numel(xTrain)/miniBatchSize/4);

layers = [...
    sequenceInputLayer(inputSize)
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
validTest_num = trainLoop_num;
for i = 1:test_num
    eval([['nmse',num2str(i),'_mat'],'= zeros(1,looptime);']);
end
for i = 1:validTest_num
    eval([['nmse_valid',num2str(i),'_mat'],'= zeros(1,looptime);']);
end

xTrain_reshape = reshape(xTrain,[],validTest_num);
yTrain_reshape = reshape(yTrain,[],validTest_num);
for i = 1:validTest_num
    xTrain_i = ['xValid',num2str(i)];
    eval([xTrain_i ,'= xTrain_reshape(:,i);']);
    yTrain_i = ['yValid',num2str(i)];
    eval([yTrain_i ,'= yTrain_reshape(:,i);']);
end

for i = 1:looptime    
    net = trainNetwork(xTrain,yTrain,layers,options);
    h = net.Layers(2).Weights.';
    for j = 1:test_num
        x_fortest = eval(['xTest',num2str(j)]);
        y_fortest = eval(['yTest',num2str(j)]);

        y_hat = predict(net,x_fortest,'MiniBatchSize',miniBatchSize);
        y_hatT = y_hat.';

        nmseNum = cellfun(@(hat,exp)10*log10(sum(sum((hat-exp).^2))/sum(sum(exp.^2))),y_hatT,y_fortest);
        nmse_loop = mean(nmseNum);
        eval([['nmse',num2str(j),'_mat(',num2str(i),')'],'=nmse_loop;']);
    end   

    for j = 1:validTest_num
        x_valid = eval(['xValid',num2str(j)]).';
        y_valid = eval(['yValid',num2str(j)]).';

        y_hat_valid = predict(net,x_valid,'MiniBatchSize',miniBatchSize);
        y_hatT_valid = y_hat_valid.';

        nmseNum_valid = cellfun(@(hat,exp)10*log10(sum(sum((hat-exp).^2))/sum(sum(exp.^2))),y_hatT_valid ,y_valid);
        nmse_loop_valid = mean(nmseNum_valid);
        eval([['nmse_valid',num2str(j),'_mat(',num2str(i),')'],'=nmse_loop_valid;']);
    end

    fprintf("already training %d times \n",i);
end
nmse_mean = zeros(1,test_num);
nmse_valid_mean = zeros(1,validTest_num);
for i = 1:test_num
    nmse_mean_tem = mean(eval(['nmse',num2str(i),'_mat']));
    nmse_mean(i) = nmse_mean_tem;
end
for i = 1:validTest_num
    nmse_valid_mean_tem = mean(eval(['nmse_valid',num2str(i),'_mat']));
    nmse_valid_mean(i) = nmse_valid_mean_tem;
end

%% Save data
savePath_txt = save_path + "/result/"+t.Month+"."+t.Day+"/10M/rand_bias"+bias+"/max_amp/Threenonlinear"+ver;   
savePath_mat = save_path + "/result/"+t.Month+"."+t.Day+"/10M/rand_bias"+bias+"/max_amp/Threenonlinear"+ver; 
if(~exist(savePath_txt,'dir'))
    mkdir(char(savePath_txt));
end
if(~exist(savePath_mat,'dir'))
    mkdir(char(savePath_mat));
end
save_parameter = fopen(savePath_txt+"/save_parameter.txt",'w');
fprintf(save_parameter,"\n \n");
fprintf(save_parameter," Threenonlinear ,\r\n ini learningRate = %e ,\r\n min batch size = %d , \r\n DropPeriod = %d ,\r\n DropFactor = %f ,\n ",...
                                      inilearningRate, miniBatchSize, LearnRateDropPeriod, LearnRateDropFactor);
fprintf(save_parameter,"train begin = %d , train end = %d , train step = %d , train data num = %d \n ", ...
    train_begin, train_end, loop_step , trainData_num);
fprintf(save_parameter,"test begin = %d , test end = %d , test step = %d , test data num = %d \n ", ...
    test_begin, test_end, loop_step , testData_num);
fprintf(save_parameter,"validationFrequency is floor(numel(xTrain)/miniBatchSize/4) \n");
fprintf(save_parameter," H order = %d \n",h_order);
fprintf(save_parameter," Hidden Units = %d \n",numHiddenUnits);
fclose(save_parameter);

save_nmse_name = 'save_nmse';
eval([save_nmse_name,'=nmse_mean;']);
save(savePath_mat+"/save_Nmse.mat",save_nmse_name);
save_nmse_valid_name = 'save_nmse_valid';
eval([save_nmse_valid_name,'=nmse_valid_mean;']);
save(savePath_mat+"/save_Nmse_valid.mat",save_nmse_valid_name);
for i = 1:test_num
    if i == 1
        save_Nmse = fopen(savePath_txt+"/save_Nmse.txt",'w');
        save_TestBandpower = fopen(savePath_txt+"/save_TestBandpower.txt",'w');
        save_TestAmp = fopen(savePath_txt+"/save_TestAmp.txt",'w');
        save_TestAmp_log = fopen(savePath_txt+"/save_TestAmp_log.txt",'w');
%         save(savePath_mat+"/save_Nmse.mat",save_nmse_name);
    else
        save_Nmse = fopen(savePath_txt+"/save_Nmse.txt",'a');
        save_TestBandpower = fopen(savePath_txt+"/save_TestBandpower.txt",'a');
        save_TestAmp = fopen(savePath_txt+"/save_TestAmp.txt",'a');
        save_TestAmp_log = fopen(savePath_txt+"/save_TestAmp_log.txt",'a');
%         save(savePath_mat+"/save_Nmse.mat",save_nmse_name,'-append');
    end
    fprintf(save_Nmse,"%f \n" , nmse_mean(i));
    fprintf(save_TestBandpower,"%f \n" , Test_bandpower(i));
    fprintf(save_TestAmp,"%f \n" , save_Testamp(i));
    fprintf(save_TestAmp_log,"%f \n" , save_Testamp_log(i));
    fclose(save_Nmse);
    fclose(save_TestBandpower);
    fclose(save_TestAmp);
    fclose(save_TestAmp_log);
end
for i = 1:validTest_num
    if i == 1
        save_Nmse_valid = fopen(savePath_txt+"/save_Nmse_valid.txt",'w');
        save_TrainBandpower = fopen(savePath_txt+"/save_TrainBandpower.txt",'w');
        save_TrainAmp = fopen(savePath_txt+"/save_TrainAmp.txt",'w');
        save_TrainAmp_log = fopen(savePath_txt+"/save_TrainAmp_log.txt",'w');
%         save(savePath_mat+"/save_Nmse.mat",save_nmse_name);
    else
        save_Nmse_valid = fopen(savePath_txt+"/save_Nmse_valid.txt",'w');
        save_TrainBandpower = fopen(savePath_txt+"/save_TrainBandpower.txt",'a');
        save_TrainAmp = fopen(savePath_txt+"/save_TrainAmp.txt",'a');
        save_TrainAmp_log = fopen(savePath_txt+"/save_TrainAmp_log.txt",'a');
%         save(savePath_mat+"/save_Nmse.mat",save_nmse_name,'-append');
    end
    fprintf(save_Nmse_valid,"%f \n" , nmse_valid_mean(i));
    fprintf(save_TrainBandpower,"%f \n" , Train_bandpower(i));
    fprintf(save_TrainAmp,"%f \n" , save_Trainamp(i));
    fprintf(save_TrainAmp_log,"%f \n" , save_Trainamp_log(i));
    fclose(save_Nmse_valid);
    fclose(save_TrainBandpower);
    fclose(save_TrainAmp);
    fclose(save_TrainAmp_log);
end

fprintf(" \n Threenonlinear , ini learningRate = %e , min batch size = %d , DropPeriod = %d , DropFactor = %f , data_num = %d \n",...
    inilearningRate, miniBatchSize, LearnRateDropPeriod, LearnRateDropFactor, data_num);
    
