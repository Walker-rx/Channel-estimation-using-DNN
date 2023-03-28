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

ver = 3;

%% Loop parameter settings
data_type = 3;
net_type = 3;
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

    data_num = 200;
    train_percent = 0.5;

    folder = '3.22';
    save_path = "data_save/light_data_"+folder;
    load_path_tmp = save_path + "/data"+data_type;
    data_path = save_path + "/data"+data_type+"/10M";
    net_path = save_path + "/result"+net_type+"/3.23/10M/mix_bias/mix_amp/Threenonlinear1";
    savePath_txt = save_path + "/result3/"+t.Month+"."+t.Day+"/10M/trainedNet/v"+ver;   
    savePath_mat = save_path + "/result3/"+t.Month+"."+t.Day+"/10M/trainedNet/v"+ver;
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

    data_num = 200;
    train_percent = 0.5;

    folder = '3.22';
    save_path = "data_save/light_data_"+folder;
    load_path_tmp = save_path + "/data"+data_type;
    data_path = save_path + "/data"+data_type+"/10M";
    net_path = save_path + "/result"+net_type+"/3.23/10M/mix_bias/mix_amp/Threenonlinear1";
    savePath_txt = save_path + "/result3/"+t.Month+"."+t.Day+"/10M/trainedNet/v"+ver;
    savePath_mat = save_path + "/result3/"+t.Month+"."+t.Day+"/10M/trainedNet/v"+ver;
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
    net_path = "data_save/light_data_3.22/result"+net_type+"/3.23/10M/mix_bias/mix_amp/Threenonlinear1";
    savePath_txt = "data_save/light_data_3.22/result3/"+t.Month+"."+t.Day+"/10M/trainedNet/v"+ver;
    savePath_mat = "data_save/light_data_3.22/result3/"+t.Month+"."+t.Day+"/10M/trainedNet/v"+ver;
end

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
        load_path = load_path_tmp+"/10M/bias"+bias+"/amp"+loop+"/mat";
        fprintf(" bias = %f , load amp = %d \n",bias,loop);
        load_data
        totalNum = data_num*split_num;
        trainNum = totalNum*train_percent;
        xTrain_tmp = x(1:trainNum);
        yTrain_tmp = y(1:trainNum);
        xTest_amp_tmp = x(trainNum+1:end);
        yTest_amp_tmp = y(trainNum+1:end);

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
        band_power((bias_loop-1)*amp_loop_num+i) = bandpower(xTrain_amp{1+(i-1)*trainNum});
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

%% Test performance with trained networks
looptime = 1;
load(net_path+"/net.mat");
for i = 1:test_num
    eval([['nmse',num2str(i),'_mat'],'= zeros(1,looptime);']);
end

for i = 1:looptime    
    for j = 1:test_num
        x_fortest = eval(['xTest',num2str(j)]);
        y_fortest = eval(['yTest',num2str(j)]);

        y_hat = predict(net,x_fortest,'MiniBatchSize',miniBatchSize);
        y_hatT = y_hat.';

        nmseNum = cellfun(@(hat,exp)10*log10(sum(sum((hat-exp).^2))/sum(sum(exp.^2))),y_hatT,y_fortest);
        nmse_loop = mean(nmseNum);
        eval([['nmse',num2str(j),'_mat(',num2str(i),')'],'=nmse_loop;']);

        figure
        plot(y_fortest{6}(6,10:35))
        hold 
        plot(y_hatT{6}(6,10:35))
        pause(6)
        close all
       
    end       
    fprintf("already predicting %d times \n",i);
end
nmse_mean = zeros(1,test_num);
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
        save_bandpower = fopen(savePath_txt+"/save_bandpower.txt",'w');
        save_amp_txt = fopen(savePath_txt+"/save_amp.txt",'w');
%         save(savePath_mat+"/save_Nmse.mat",save_nmse_name);
    else
        save_Nmse = fopen(savePath_txt+"/save_Nmse.txt",'a');
        save_bandpower = fopen(savePath_txt+"/save_bandpower.txt",'a');
        save_amp_txt = fopen(savePath_txt+"/save_amp.txt",'a');
%         save(savePath_mat+"/save_Nmse.mat",save_nmse_name,'-append');
    end
    fprintf(save_Nmse,"%f \n" , nmse_mean(i));
    fprintf(save_bandpower,"%f \n" , band_power(i));
    fprintf(save_amp_txt,"%f \n" , save_amp(i));
    fclose(save_Nmse);
    fclose(save_bandpower);
    fclose(save_amp_txt);
end
fprintf(" result saved in %s \n",savePath_mat);

    
