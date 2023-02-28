clear 
close all

save_path = "data_save/2.26";
snr_begin = -6;
snr_end = 42;
train_order = 1;
test_order = 9;
% snr = 2;
fprintf("v1 \n");
for snr = 50:4:50
    amp = 33;
%     load_path = save_path + "/25M/8pam/amp"+amp+"/mat";
%     fprintf("amp = %d \n",amp);
%     pilot_length = 2047;
%     zero_length = 3000;
%     x_mat = load(load_path+"/save_signal_ori.mat");
%     y_mat = load(load_path+"/save_signal_received_real_send1.mat");
%     fin_mat = load(load_path+"/save_fin_syn_point_real_send1.mat");
%     x_names = fieldnames(x_mat);
%     y_names = fieldnames(y_mat);
%     fin_names = fieldnames(fin_mat);
%     data_num = 200;
%     
%     x = cell(1,data_num*10);
%     y = cell(1,data_num*10);
%     for name_order = 1:data_num
%         signal_ori = gather(eval(strcat('x_mat.',x_names{1})));
%         signal_received = gather(eval(strcat('y_mat.',y_names{1})));
%         fin_syn_point = gather(eval(strcat('fin_mat.',fin_names{1})));
%         data_ori = signal_ori(pilot_length+zero_length+1:end);
%         data_received = signal_received(fin_syn_point + (pilot_length+zero_length)*6 : end);
%         for i = 1:10
%             x{10*(name_order-1)+i} = [zeros(1,15),data_ori(1000*(i-1)+1:1000*i)];
%             y{10*(name_order-1)+i} = data_received(6000*(i-1)+1:6000*i);
%         end
%     end
%     x = gather(x);
%     y = gather(y);
%     clear x_mat y_mat x_names y_names
    
    load_path = save_path + "/data/snr"+snr;
    load_data
    x = cellfun(@(cell1)(cell1*100*1.1^amp),x,'UniformOutput',false);
    xTrain = x(1);
    yTrain = y(1);
    xTest = x(2:end);
    yTest = y(2:end);

    h_order = 30;
    h = zeros(30,6);

    xTrain{1} = [zeros(1,10) xTrain{1}.'];
    xTrain{1} = toeplitz(xTrain{1}(h_order:-1:1),xTrain{1}(h_order:end)).';
%     for i = 1:6
%         yTrain2 = yTrain{1}(i:6:i+6*(size(xTrain{1},1)-1));
%         h_hat = (xTrain{1}'*xTrain{1})\xTrain{1}'*yTrain2.';
%         h(:,i) = h_hat;
%     end
%     yTrain{1} = yTrain{1}.';
    yTrain{1} = yTrain{1}(1:size(xTrain{1},1));
    yTrain2 = yTrain{1};
    h_hat = (xTrain{1}'*xTrain{1})\xTrain{1}'*yTrain2;

    Mse = zeros(1,numel(xTest));
    for j = 1:numel(xTest)
        xTest{j} = [zeros(1,10) xTest{j}.'];
        xTest{j} = toeplitz(xTest{j}(h_order:-1:1),xTest{j}(h_order:end)).';      
        xTest2 = xTest{j};       
        yTest{j} = yTest{j}(1:size(xTest{j},1));
        yTest2 = yTest{j};
        y_hat = xTest2*h_hat;

        Msei = mse(y_hat,yTest2);
        Mse(1,j) = Msei;
    end
    mmse = mean(Mse);
    fprintf("%e \n",mmse);

%     Mse = zeros(numel(xTest),6);
%     for j = 1:numel(xTest)
%         xTest{j} = [zeros(1,10) xTest{j}.'];
%         xTest{j} = toeplitz(xTest{j}(h_order:-1:1),xTest{j}(h_order:end)).';      
%         xTest2 = xTest{j};
%         for k = 1:6
% %             k = 6;
%             yTest2 = yTest{j}(k:6:k+6*(length(xTest2)-1));
%             y_hat = xTest2*h(:,k);
%             Msei = mse(y_hat,yTest2.');
%             Mse(j,k) = Msei;
%         end
%     end
%     mmse = mean(mean(Mse));
%     fprintf("%e \n",mmse);


    savePath_result = save_path + "/result/ls/snr" + snr;
    if(~exist(savePath_result,'dir'))
        mkdir(char(savePath_result));
    end
    
    saveH = 'save_h';
    eval([saveH,'=h;']);
    save(savePath_result+"/save_h.mat",saveH);

    if snr == snr_begin
        save_snr = fopen(save_path+"/result/ls/save_snr.txt",'w');
        save_Mse = fopen(save_path+"/result/ls/save_Mse.txt",'w');
    else
        save_snr = fopen(save_path+"/result/ls/save_snr.txt",'a');
        save_Mse = fopen(save_path+"/result/ls/save_Mse.txt",'a');
    end  
    fprintf(save_snr,'%d \r\n',snr);
    fprintf(save_Mse,'%.6g \r\n',mmse);
    fclose(save_snr);
    fclose(save_Mse);
    fprintf(' snr = %d , mse = %.6g \r\n',snr,mmse);
end
