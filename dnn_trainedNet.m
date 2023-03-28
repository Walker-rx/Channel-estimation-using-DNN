clear
close all

t = datetime('now');

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
miniBatchSize = 400;
LearnRateDropPeriod = 8;
LearnRateDropFactor = 0.1;
inilearningRate = 1e-2;

ver = 1;

%% Loop parameter settings
data_type = 1;
net_type = 1;
if data_type == 1
    bias_begin = 0.05;
    bias_step = 0.04;
    bias_end = 0.85;
    bias_scope = 0.05:0.04:0.85;
    bias_loop_num = (bias_end-bias_begin)/bias_step+1;

    amp = 0.1613;
    amp_loop_begin = amp;
    amp_loop_end = amp;
    amp_loop_step = 1;
    amp_loop_num = (amp_loop_end - amp_loop_begin)/amp_loop_step + 1 ;

    amp_begin = amp;
    amp_norm = 0;

    data_num = 200;
    train_percent = 0.5;

    folder = '3.22';
    save_path = "data_save/light_data_"+folder;
    data_path = save_path + "/data/10M/amp"+amp;
    net_path = save_path + "/result4/3.27/mix_bias_amp/Threenonlinear1/net/net"+net_type;
    savePath_txt = save_path + "/result3/"+t.Month+"."+t.Day+"/trainedNet/v"+ver;
    savePath_mat = save_path + "/result3/"+t.Month+"."+t.Day+"/trainedNet/v"+ver;
elseif data_type == 2
    bias_begin = 0.05;
    bias_step = 0.04;
    bias_end = 0.85;
    bias_scope = 0.05:0.04:0.85;
    bias_loop_num = (bias_end-bias_begin)/bias_step+1;

    amp = 1;
    amp_loop_begin = amp;
    amp_loop_end = amp;
    amp_loop_step = 1;
    amp_loop_num = (amp_loop_end - amp_loop_begin)/amp_loop_step + 1 ;

    amp_begin = amp;
    amp_norm = 0;

    data_num = 200;
    train_percent = 0.5;

    folder = '3.22';
    save_path = "data_save/light_data_"+folder;
    data_path = save_path + "/data/10M/amp"+amp;
    net_path = save_path + "/result4/3.27/mix_bias_amp/Threenonlinear1/net/net"+net_type;
    savePath_txt = save_path + "/result3/"+t.Month+"."+t.Day+"/trainedNet/v"+ver;
    savePath_mat = save_path + "/result3/"+t.Month+"."+t.Day+"/trainedNet/v"+ver;
elseif data_type == 3
    bias_begin = 0.3;
    bias_step = 1;
    bias_end = 0.3;
    bias_loop_num = (bias_end-bias_begin)/bias_step+1;

    amp_loop_begin = 1;
    amp_loop_end = 26;
    amp_loop_step = 1;
    amp_loop_num = (amp_loop_end - amp_loop_begin)/amp_loop_step + 1 ;

    amp_begin = 0.0015;
    amp_norm = 0.03994;

    data_num = 100;
    train_percent = 0.005;

    folder = '3.17';
    save_path = "data_save/light_data_"+folder;
    load_path_tmp = save_path + "/data";
    data_path = save_path + "/data/10M";
    net_path = "data_save/light_data_3.22/result4/3.27/mix_bias_amp/Threenonlinear1/net"+net_type;
    savePath_txt = "data_save/light_data_3.22/result3/"+t.Month+"."+t.Day+"/10M/trainedNet/v"+ver;
    savePath_mat = "data_save/light_data_3.22/result3/"+t.Month+"."+t.Day+"/10M/trainedNet/v"+ver;
end

%% Load data
test_num_amp = 0;
bias_loop = 0;
xTrain = [];
yTrain = [];
test_num = bias_loop_num*amp_loop_num;
test_num = round(test_num);
save_amp = zeros(1,round(test_num));
save_bias = zeros(1,round(test_num));
band_power = zeros(1,round(amp_loop_num));

for loop = amp_loop_begin : amp_loop_step :amp_loop_end
    test_num_amp = test_num_amp +1;
    test_num_bias = 0;
    for bias = bias_begin : bias_step : bias_end
        test_num_bias = test_num_bias + 1;
        load_path = data_path+"/bias"+bias+"/mat";
        fprintf(" load amp = %f ,bias = %f  \n",amp,bias);
        load_data
        totalNum = data_num*split_num;
        trainNum = totalNum*train_percent;
        xTrain_tmp = x(1:trainNum);
        yTrain_tmp = y(1:trainNum);
        xTest_bias_tmp = x(trainNum+1:end);
        yTest_bias_tmp = y(trainNum+1:end);

        amp_loop = 32000*(amp_begin+(test_num_bias-1)*amp_norm);
        save_amp((test_num_amp-1)*bias_loop_num+test_num_bias) = 10*log10(amp_loop^2);
        save_bias((test_num_amp-1)*bias_loop_num+test_num_bias) = bias;
        xTrain_tmp = cellfun(@(cell1)(cell1*amp_loop),xTrain_tmp,'UniformOutput',false);
        xTest_bias_tmp = cellfun(@(cell1)(cell1*amp_loop),xTest_bias_tmp,'UniformOutput',false);

        xTest_bias_name = ['xTest_',num2str((test_num_amp-1)*bias_loop_num+test_num_bias)];
        yTest_bias_name = ['yTest_',num2str((test_num_amp-1)*bias_loop_num+test_num_bias)];
        eval([xTest_bias_name,'=xTest_bias_tmp;']);
        eval([yTest_bias_name,'=yTest_bias_tmp;']);

        xTrain = [xTrain xTrain_tmp];
        yTrain = [yTrain yTrain_tmp];

        clear x y
    end   
end
%%  Normalize data
totaltrain = numel(xTrain);
% norm_cell = xTrain{floor(totaltrain/2)};
% norm_factor = 1/norm(norm_cell)*sqrt(length(norm_cell));
% load_path = "data_save/light_data_2.28/result/3.1/25M/8pam/mix_amp/Twononlinear";

load_path = "data_save/light_data_3.10/data/10M/rand_bias0.3/";
norm_mat = load(load_path+"/save_norm.mat");
norm_names = fieldnames(norm_mat);
norm_factor = gather(eval(strcat('norm_mat.',norm_names{1})));

xTrain = cellfun(@(cell1)(cell1*norm_factor),xTrain,'UniformOutput',false);

for i = 1:amp_loop_num
    band_power(i) = bandpower(xTrain{ (i-1)*bias_loop_num+10 });
end

for i = 1:test_num
    xTest_nor = eval(['xTest_',num2str(i)]);
    xTest_nor = cellfun(@(cell1)(cell1*norm_factor),xTest_nor,'UniformOutput',false);
    eval([['xTest_',num2str(i)],'= xTest_nor;']);
end

%%  Reshape data
for i = 1:numel(xTrain)
    xTrain{i} = toeplitz(xTrain{i}(h_order:-1:1),xTrain{i}(h_order:end));
    xTrain{i} = [xTrain{i}; bias_scope( floor((i-1)/trainNum)+1 )*ones(1,size(xTrain{i},2) )];
    yTrain{i} = reshape(yTrain{i}(1:split_length*rate_times),outputSize,split_length);
    yTrain{i} = yTrain{i}(:,1:size(xTrain{i},2));
end
for i = 1:test_num
    xtop_tem = eval(['xTest_',num2str(i)]);
    ytop_tem = eval(['yTest_',num2str(i)]);
    for j = 1:numel(xtop_tem)
        xtop_tem{j} = toeplitz(xtop_tem{j}(h_order:-1:1),xtop_tem{j}(h_order:end));
        xtop_tem{j} = [xtop_tem{j}; bias_scope( floor((i-1)/trainNum)+1 )*ones(1,size(xtop_tem{j},2) )];
        ytop_tem{j} = reshape(ytop_tem{j}(1:split_length*rate_times),outputSize,split_length);
        ytop_tem{j} = ytop_tem{j}(:,1:size(xtop_tem{j},2));
    end
    eval([['xTest_',num2str(i)],'= xtop_tem;']);
    eval([['yTest_',num2str(i)],'= ytop_tem;']);
end

%% Test performance with trained networks
looptime = 2;
load(net_path+"/net.mat");
for i = 1:test_num
    eval([['nmse',num2str(i),'_mat'],'= zeros(1,looptime);']);
end

for i = 1:looptime    
    for j = 1:test_num
        x_fortest = eval(['xTest_',num2str(j)]);
        y_fortest = eval(['yTest_',num2str(j)]);

        y_hat = predict(net,x_fortest,'MiniBatchSize',miniBatchSize);
        y_hatT = y_hat.';

        nmseNum = cellfun(@(hat,exp)10*log10(sum(sum((hat-exp).^2))/sum(sum(exp.^2))),y_hatT,y_fortest);
        nmse_loop = mean(nmseNum);
        eval([['nmse',num2str(j),'_mat(',num2str(i),')'],'=nmse_loop;']);

        figure
        plot(y_fortest{6}(6,10:35))
        hold 
        plot(y_hatT{6}(6,10:35))
        pause(3)
        close all       
    end       
    fprintf("already predicting %d loop times \n",i);
end
nmse_mean = zeros(1,round(test_num));
for i = 1:test_num
    nmse_mean_tem = mean(eval(['nmse',num2str(i),'_mat']));
    nmse_mean(i) = nmse_mean_tem;
end

%% Save data

if(~exist(savePath_txt,'dir'))
    mkdir(char(savePath_txt));
end
if(~exist(savePath_mat,'dir'))
    mkdir(char(savePath_mat));
end
save_parameter = fopen(savePath_txt+"/save_parameter.txt",'w');
fprintf(save_parameter,"\n \n");
fprintf(save_parameter," The network used is %s \r\n " , net_path);
fprintf(save_parameter,"The data used is %s \r\n " , data_path);
fclose(save_parameter);

save_nmse_name = 'save_nmse';
eval([save_nmse_name,'=nmse_mean;']);
save(savePath_mat+"/save_Nmse.mat",save_nmse_name);
 
for i = 1:test_num
    if i == 1
        save_Nmse = fopen(savePath_txt+"/save_Nmse.txt",'w');
        save_amp_bias_txt = fopen(savePath_txt+"/save_amp.txt",'w');
%         save(savePath_mat+"/save_Nmse.mat",save_nmse_name);
    else
        save_Nmse = fopen(savePath_txt+"/save_Nmse.txt",'a');
        save_amp_bias_txt = fopen(savePath_txt+"/save_amp.txt",'a');
%         save(savePath_mat+"/save_Nmse.mat",save_nmse_name,'-append');
    end
    fprintf(save_Nmse,"%f \n" , nmse_mean(i));
    fprintf(save_amp_bias_txt," amp = %f ,bias = %f \n" , save_amp(i),save_bias(i));
    fclose(save_Nmse);
    fclose(save_amp_bias_txt);
end

for i = 1:amp_loop_num
    if i == 1
        save_bandpower = fopen(savePath_txt+"/save_bandpower.txt",'w');
    else
        save_bandpower = fopen(savePath_txt+"/save_bandpower.txt",'a');
    end
    fprintf(save_bandpower,"%f \n" , band_power(i));
end

fprintf(" result saved in %s \n",savePath_mat);

    
