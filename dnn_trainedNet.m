clear
close all
gpuDevice(1)

t = datetime('now');

%% Network parameters
ori_rate = 30e6;
rec_rate = 60e6;
rate_times = rec_rate/ori_rate;
h_order = 30;
add_zero = h_order/2;
 
split_num = 10;  % Cut a signal into split_num shares

inputSize = h_order+1;
numHiddenUnits = 60;
outputSize = rate_times;  % y=h*x+n;  y:(outputSize,m) h:(outputSize,inputSize) x:(inputSize,m)
miniBatchSize = 40;

bias_begin = 0.05;
bias_step = 0.04;
bias_end = 1.05;
bias_scope = bias_begin : bias_step : bias_end;
bias_loop_num = (bias_end-bias_begin)/bias_step+1;
bias_loop_num = round(bias_loop_num);

% amp_scope = [0.1613 0.32106 0.48082 0.64058 0.8003 1];
amp_scope = [0.005 0.007 0.015 0.024 0.034 0.045 0.08 0.18 0.25 0.3 0.48082 0.64058 0.8003 1];
amp_loop_num = numel(amp_scope) ;

load_begin = 61;
load_end = 80;
data_num = load_end-load_begin+1;
train_percent = 0.05;

save_folder = '6.18';
data_folder = '6.18';
net_folder = '6.20';

save_path = "data_save/light_data_"+save_folder;
data_path_ini = "/home/oem/Users/ruoxu/equalization-using-DNN/data_save/light_data_"+data_folder;

ver = 3;

%% Loop parameter settings
total_ver = 20;
for net_ver = 1:total_ver
    net_type = [ver,1,net_ver];
    net_path = save_path + "/result1/"+net_folder+"/mix_bias_amp/Threenonlinear"+net_type(1)+"/net/looptime"+net_type(2)+"/net"+net_type(3);
    
    test_num_amp = 0;    
    band_power = zeros(1,round(amp_loop_num));

    for amp_loop = 1:amp_loop_num
        %% Load data
        xTrain = [];
        yTrain = [];
        amp = amp_scope(amp_loop);
        save_amp = zeros(1,round(bias_loop_num));
        save_bias = zeros(1,round(bias_loop_num));


        data_path = data_path_ini + "/data/"+ori_rate/1e6+"M/amp"+amp;
        
        savePath_txt = save_path + "/result3/"+t.Month+"."+t.Day+"/trainedNet/v"+ver+"/amp"+amp+"/net"+net_ver;
        savePath_mat = save_path + "/result3/"+t.Month+"."+t.Day+"/trainedNet/v"+ver+"/amp"+amp+"/net"+net_ver;

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

            amp_nor = 32000*amp;
            save_amp(test_num_bias) = 10*log10(amp_nor^2);
            save_bias(test_num_bias) = bias;
            xTrain_tmp = cellfun(@(cell1)(cell1*amp_nor),xTrain_tmp,'UniformOutput',false);
            xTest_bias_tmp = cellfun(@(cell1)(cell1*amp_nor),xTest_bias_tmp,'UniformOutput',false);

            xTest_bias_name = ['xTest_',num2str(test_num_bias)];
            yTest_bias_name = ['yTest_',num2str(test_num_bias)];
            eval([xTest_bias_name,'=xTest_bias_tmp;']);
            eval([yTest_bias_name,'=yTest_bias_tmp;']);

            xTrain = [xTrain xTrain_tmp];
            yTrain = [yTrain yTrain_tmp];

            clear x y
        end  
        
        %%  Normalize data

        % load_path = "data_save/light_data_3.10/data/10M/rand_bias0.3/";
        load_norm_path = "/home/oem/Users/ruoxu/channel-estimation-using-DNN/data_save/norm_factor/";
        norm_mat = load(load_norm_path+"/save_norm.mat");
        norm_names = fieldnames(norm_mat);
        norm_factor = gather(eval(strcat('norm_mat.',norm_names{1})));

        xTrain = cellfun(@(cell1)(cell1*norm_factor),xTrain,'UniformOutput',false);
        
        band_power(amp_loop) = bandpower(xTrain{10});

        for i = 1:bias_loop_num
            xTest_nor = eval(['xTest_',num2str(i)]);
            xTest_nor = cellfun(@(cell1)(cell1*norm_factor),xTest_nor,'UniformOutput',false);
            eval([['xTest_',num2str(i)],'= xTest_nor;']);
        end

        %%  Reshape data
        for i = 1:numel(xTrain)
            xTrain{i} = toeplitz(xTrain{i}(h_order:-1:1),xTrain{i}(h_order:end));
            xTrain{i} = [xTrain{i}; single( bias_scope( floor((i-1)/trainNum)+1 ) )*ones(1,size(xTrain{i},2) )];
            yTrain{i} = reshape(yTrain{i}(1:split_length*rate_times),outputSize,split_length);
            yTrain{i} = yTrain{i}(:,1:size(xTrain{i},2));
        end
        for i = 1:bias_loop_num
            xtop_tem = eval(['xTest_',num2str(i)]);
            ytop_tem = eval(['yTest_',num2str(i)]);
            for j = 1:numel(xtop_tem)
                xtop_tem{j} = toeplitz(xtop_tem{j}(h_order:-1:1),xtop_tem{j}(h_order:end));
                xtop_tem{j} = [xtop_tem{j}; single( bias_scope( i ) )*ones(1,size(xtop_tem{j},2) )];
                ytop_tem{j} = reshape(ytop_tem{j}(1:split_length*rate_times),outputSize,split_length);
                ytop_tem{j} = ytop_tem{j}(:,1:size(xtop_tem{j},2));
            end
            eval([['xTest_',num2str(i)],'= xtop_tem;']);
            eval([['yTest_',num2str(i)],'= ytop_tem;']);
        end

        %% Test performance with trained networks
        looptime = 2;
        load(net_path+"/net.mat");
        fprintf(" load net = %s \n",net_path);
        for i = 1:bias_loop_num
            eval([['nmse',num2str(i),'_mat'],'= zeros(1,looptime);']);
        end

        for i = 1:looptime    
            for j = 1:bias_loop_num
                x_fortest = eval(['xTest_',num2str(j)]);
                y_fortest = eval(['yTest_',num2str(j)]);
        %% Custom network
                % x_fortest = cell2mat(x_fortest);
                % y_fortest = cell2mat(y_fortest);
                % 
                % x_fortest(:,:,1) = x_fortest;
                % y_fortest(:,:,1) = y_fortest;
                % x_fortest = dlarray(single(x_fortest),'CBT');
                % y_fortest = dlarray(single(y_fortest),'CBT');
                % x_fortest = gpuArray(x_fortest);
                % y_fortest = gpuArray(y_fortest);
                % miniBatchSize = dlarray(miniBatchSize);
                % 
                % y_hat = predict(dlnet,x_fortest);
                % 
                % y_fortest = extractdata(y_fortest);
                % y_hat = extractdata(y_hat);
                % y_fortest = gather(y_fortest);
                % y_hat = gather(y_hat);
                % y_fortest = double(y_fortest);
                % y_hat = double(y_hat);
                % y_hatT = y_hat;
                % nmseNum_fun = @(hat,exp)10*log10(sum(sum((hat-exp).^2))/sum(sum(exp.^2)));
                % nmseNum = nmseNum_fun(y_hatT,y_fortest);

        %% Default network
                y_hat = predict(net,x_fortest,'MiniBatchSize',miniBatchSize);
                y_hatT = y_hat.';
                nmseNum = cellfun(@(hat,exp)10*log10(sum(sum((hat-exp).^2))/sum(sum(exp.^2))),y_hatT,y_fortest);
                
        %%
                nmse_loop = mean(nmseNum);
                eval([['nmse',num2str(j),'_mat(',num2str(i),')'],'=nmse_loop;']);

%                 figure
%                 plot(y_fortest{6}(6,10:35))
%                 hold 
%                 plot(y_hatT{6}(6,10:35))
%                 figure
%                 plot(y_fortest(6,10:35))
%                 hold 
%                 plot(y_hatT(6,10:35))
                fprintf(" amp = %d/%d , net = v%d/%d, looptime = %d/%d , bias num = %d/%d \n",amp_loop,amp_loop_num,net_ver,total_ver,i,looptime,j,bias_loop_num);
                pause(3)
                close all       
            end       
        end
        nmse_mean = zeros(1,round(bias_loop_num));
        for i = 1:bias_loop_num
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

        for i = 1:bias_loop_num
            if i == 1
                save_Nmse = fopen(savePath_txt+"/save_Nmse.txt",'w');
                save_amp_bias_txt = fopen(savePath_txt+"/save_amp.txt",'w');
            else
                save_Nmse = fopen(savePath_txt+"/save_Nmse.txt",'a');
                save_amp_bias_txt = fopen(savePath_txt+"/save_amp.txt",'a');
            end
            fprintf(save_Nmse,"%f \n" , nmse_mean(i));
            fprintf(save_amp_bias_txt," amp = %f ,bias = %f \n" , save_amp(i),save_bias(i));
            fclose(save_Nmse);
            fclose(save_amp_bias_txt);
        end

        save_bandpower = fopen(savePath_txt+"/save_bandpower.txt",'w');
        fprintf(save_bandpower,"%f \n" , band_power(amp_loop));
        fclose(save_bandpower);

        fprintf(" result saved in %s \n",savePath_mat);
    end

end

    
