clear 
close all

save_path = "data_save/line_data";
snr_begin = 2;
snr_end = 50;
train_order = 1;
test_order = 10;
% snr = 2;
for snr = snr_begin:4:snr_end
%     clearvars -except snr save_path snr_begin snr_end
    load_path = save_path + "/snr"+snr;

    x_mat = load(load_path+"/save_x.mat");
    y_mat = load(load_path+"/save_y.mat");
    x_names = fieldnames(x_mat);
    y_names = fieldnames(y_mat);

    xTrain = eval(strcat('x_mat.',x_names{train_order}));
    yTrain = eval(strcat('y_mat.',y_names{train_order}));
    xTest = eval(strcat('x_mat.',x_names{test_order}));
    yTest = eval(strcat('y_mat.',y_names{test_order}));

    clear x_mat y_mat x_names y_names
    
    h_order = length(yTrain) - length(xTrain) + 1;
    xTrain = toeplitz(xTrain(h_order:-1:1),xTrain(h_order:end)).';
    xTest = toeplitz(xTest(h_order:-1:1),xTest(h_order:end)).';
    yTrain = yTrain(h_order:length(yTrain)-(h_order-1));
    yTest = yTest(h_order:length(yTest)-(h_order-1));

    h_hat = (xTrain'*xTrain)\xTrain'*yTrain;
    y_hat = xTest*h_hat;

    Mse = mse(y_hat,yTest);

    savePath_result = save_path + "/result/ls/snr" + snr;
    if(~exist(savePath_result,'dir'))
        mkdir(char(savePath_result));
    end
    
    saveH = 'save_h';
    saveyHat = 'save_y_hat';
    eval([saveH,'=h_hat;']);
    eval([saveyHat,'=y_hat;']);
    save(savePath_result+"/save_h.mat",saveH);
    save(savePath_result+"/save_yHat.mat",saveyHat);

    if snr == snr_begin
        save_snr = fopen(save_path+"/result/ls/save_snr.txt",'w');
        save_Mse = fopen(save_path+"/result/ls/save_Mse.txt",'w');
    else
        save_snr = fopen(save_path+"/result/ls/save_snr.txt",'a');
        save_Mse = fopen(save_path+"/result/ls/save_Mse.txt",'a');
    end  
    fprintf(save_snr,'%d \r\n',snr);
    fprintf(save_Mse,'%.6g \r\n',Mse);
    fclose(save_snr);
    fclose(save_Mse);
    fprintf(' snr = %d , mse = %.6g \r\n',snr,Mse);
%     clearvars -except snr save_path snr_begin snr_end
end
