clear 
close all

save_path = "data_save/2.23";
snr_begin = -6;
snr_end = 42;
train_order = 1;
test_order = 9;
% snr = 2;
fprintf("v1");
for snr = 42:4:snr_end
%     clearvars -except snr save_path snr_begin snr_end
    load_path = save_path + "/data2/snr"+snr;
    load_data
    
    xTrain = x(1);
    yTrain = y(1);
    xTest = x(2:end);
    yTest = y(2:end);

    h_order = 30;
    h = zeros(30,6);

%     xTrain{1} = [zeros(15,1); xTrain{1}];

    xTrain{1} = toeplitz(xTrain{1}(h_order:-1:1),xTrain{1}(h_order:end)).';
    for i = 1:6
        yTrain2 = yTrain{1}(i:6:i+6*(size(xTrain{1},1)-1));
        h_hat = (xTrain{1}'*xTrain{1})\xTrain{1}'*yTrain2;
        h(:,i) = h_hat;
    end

    Mse = zeros(numel(xTest),6);
    for j = 1:numel(xTest)
%         xTest{j} = [zeros(15,1); xTest{j}];
        xTest{j} = toeplitz(xTest{j}(h_order:-1:1),xTest{j}(h_order:end)).';      
        xTest2 = xTest{j};
        for k = 1:6
%             k = 6;
            yTest2 = yTest{j}(k:6:k+6*(length(xTest2)-1));
            y_hat = xTest2*h(:,k);
            Msei = mse(y_hat,yTest2);
            Mse(j,k) = Msei;
        end
    end
    mmse = mean(mean(Mse));

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
