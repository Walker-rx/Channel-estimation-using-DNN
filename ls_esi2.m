clear 
close all

t = datetime('now');
save_path = "data_save/light_data_3.10";
amp_begin = 1;
amp_end = 26;
looptime = 0;
bias = 0.3;
fprintf("v1 \n");
for amp = amp_begin:amp_end
%% Load data
    looptime = looptime + 1;
    load_path = save_path + "/data/10M/rand_bias"+bias+"/amp"+amp+"/mat";
    load_data
%% Normalize data
%     x = cellfun(@(cell1)(cell1*100*1.1^amp),x,'UniformOutput',false);
    x = cellfun(@(cell1)(cell1*32000*(0.0015+(amp-1)*0.03994)),x,'UniformOutput',false);
    load_path = "data_save/light_data_2.28/result/3.1/25M/8pam/mix_amp/Twononlinear";
    norm_mat = load(load_path+"/save_norm.mat");
    norm_names = fieldnames(norm_mat);
    norm_factor = gather(eval(strcat('norm_mat.',norm_names{1})));
    x = cellfun(@(cell1)(cell1*norm_factor),x,'UniformOutput',false);
%%
    xTrain = x(1);
    yTrain = y(1);
    xTest = x(2:end);
    yTest = y(2:end);

    h_order = 30;
    h = zeros(30,6);
%% Reshape data
%     xTrain{1} = [zeros(1,10) xTrain{1}.'];
    xTrain{1} = toeplitz(xTrain{1}(h_order:-1:1),xTrain{1}(h_order:end)).';
    for i = 1:6
        yTrain2 = yTrain{1}(i:6:i+6*(size(xTrain{1},1)-1));
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
    Nmse_mat = zeros(numel(xTest),6);
    for j = 1:numel(xTest)
%         xTest{j} = [zeros(1,10) xTest{j}.'];
        xTest{j} = toeplitz(xTest{j}(h_order:-1:1),xTest{j}(h_order:end)).';      
        xTest2 = xTest{j};
        for k = 1:6
            yTest2 = yTest{j}(k:6:k+6*(length(xTest2)-1));
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
    if amp == amp_begin
        save_Nmse = fopen(savePath_result+"/save_Nmse.txt",'w');
        save(savePath_result+"/save_h.mat",saveH);
    else
        save_Nmse = fopen(savePath_result+"/save_Nmse.txt",'a');
        save(savePath_result+"/save_h.mat",saveH,'-append');
    end  
    fprintf(save_Nmse,'%f \r\n',Nmse);
    fclose(save_Nmse);
    fprintf(' amp = %d , nmse = %.6g \r\n',amp,Nmse);
end
