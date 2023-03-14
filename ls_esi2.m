% clear 
% close all

t = datetime('now');
save_path = "data_save/light_data_3.11";
ori_rate = 10e6;
rec_rate = 150e6;
rate_times = rec_rate/ori_rate;
split_num = 1;

loop_begin = 1;
loop_end = 101;
loop_step = 2;
amp_begin = 0.0015;
amp_norm = 0.009985;
% amp_norm = 0.03994;
looptime = 0;
bias = 0.3;
fprintf("light_data_3.11 v1 \n");
for loop = loop_begin:loop_step:loop_end
%% Load data
    looptime = looptime + 1;
    load_path = save_path + "/data/10M/rand_bias"+bias+"/amp"+loop+"/mat";
    load_data
%% Normalize data
%     x = cellfun(@(cell1)(cell1*100*1.1^amp),x,'UniformOutput',false);
    x = cellfun(@(cell1)(cell1*32000*(amp_begin+(loop-1)*amp_norm)),x,'UniformOutput',false);
    load_path = "data_save/light_data_3.10/data/10M/rand_bias0.3/";
    norm_mat = load(load_path+"/save_norm.mat");
    norm_names = fieldnames(norm_mat);
    norm_factor = gather(eval(strcat('norm_mat.',norm_names{1})));
    x = cellfun(@(cell1)(cell1*norm_factor),x,'UniformOutput',false);
%%
    xTrain = x(1);
    yTrain = y(1);
    xTest = x(2:end);
    yTest = y(2:end);
    
    band_power = bandpower(xTrain{1});

    h_order = 30;
    h = zeros(30,6);
%% Reshape data
%     xTrain{1} = [zeros(1,10) xTrain{1}.'];
    xTrain{1} = toeplitz(xTrain{1}(h_order:-1:1),xTrain{1}(h_order:end)).';
    for i = 1:rate_times 
        yTrain2 = yTrain{1}(i:rate_times :i+rate_times *(size(xTrain{1},1)-1));
        h_hat = (xTrain{1}'*xTrain{1})\xTrain{1}'*yTrain2.';
        h(:,i) = h_hat;
    end
    
%% LS for same sampling rate
%     yTrain{1} = yTrain{1}.';
%     yTrain{1} = yTrain{1}(1:size(xTrain{1},1));
%     yTrain2 = yTrain{1};
%     h_hat = (xTrain{1}'*xTrain{1})\xTrain{1}'*yTrain2;
%     Mse = zeros(1,numel(xTest));
%     for j = 1:numel(xTest)
%         xTest{j} = [zeros(1,10) xTest{j}.'];
%         xTest{j} = toeplitz(xTest{j}(h_order:-1:1),xTest{j}(h_order:end)).';      
%         xTest2 = xTest{j};       
%         yTest{j} = yTest{j}(1:size(xTest{j},1));
%         yTest2 = yTest{j};
%         y_hat = xTest2*h_hat;
% 
%         Msei = mse(y_hat,yTest2);
%         Mse(1,j) = Msei;
%     end
%     mmse = mean(Mse);
%     fprintf("%e \n",mmse);

%% LS for different sampling rate
    Nmse_mat = zeros(numel(xTest),rate_times);
    for j = 1:numel(xTest)
%         xTest{j} = [zeros(1,10) xTest{j}.'];
        xTest{j} = toeplitz(xTest{j}(h_order:-1:1),xTest{j}(h_order:end)).';      
        xTest2 = xTest{j};
        for k = 1:rate_times 
            yTest2 = yTest{j}(k:rate_times :k+rate_times*(length(xTest2)-1));
            y_hat = xTest2*h(:,k);
            y_hat = y_hat.';
%             Msei = mse(y_hat,yTest2.');
            Nmsei = 10*log10(sum((y_hat-yTest2).^2)/sum(yTest2.^2));
            Nmse_mat(j,k) = Nmsei;
        end
    end
    Nmse = mean(mean(Nmse_mat));

%%  Save data
    savePath_result = save_path + "/result/"+t.Month+"."+t.Day+"/10M/rand_bias"+bias+"/norm_LS";
    if(~exist(savePath_result,'dir'))
        mkdir(char(savePath_result));
    end
    
    saveH = ['save_h_' num2str(looptime)];
    eval([saveH,'=h;']);   
    if loop == loop_begin
        save_Nmse = fopen(savePath_result+"/save_Nmse.txt",'w');
        save_bandpower = fopen(savePath_result+"/save_bandpower.txt",'w');
        save(savePath_result+"/save_h.mat",saveH);
    else
        save_Nmse = fopen(savePath_result+"/save_Nmse.txt",'a');
        save_bandpower = fopen(savePath_result+"/save_bandpower.txt",'a');
        save(savePath_result+"/save_h.mat",saveH,'-append');
    end  
    fprintf(save_Nmse,'%f \r\n',Nmse);
    fprintf(save_bandpower,'%f \r\n',band_power);
    fclose(save_Nmse);
    fclose(save_bandpower);
    fprintf(' amp = %d , nmse = %.6g \r\n',loop,Nmse);
end
