function [ ] = dnn_train_default(train_time, savePath_mat, xTrain, yTrain, layers, options, test_num,...
                                                   save_amp, bias_save, band_power, load_begin, load_end)
    if train_time<=1
        net_path = savePath_mat+"/net/looptime1/net"+train_time;
        if(~exist(net_path,'dir'))
            mkdir(char(net_path));
        end
    
        net = trainNetwork(xTrain,yTrain,layers,options);
        save(net_path+"/net.mat",'net');  % Save the trained network
    
        fprintf("already training %d times \n",train_time);

    else
        net_path = savePath_mat+"/net/looptime1/net"+train_time;
        if(~exist(net_path,'dir'))
            mkdir(char(net_path));
        end
    
        load(savePath_mat+"/net/looptime1/net"+(train_time-1)+"/net.mat");
        layers = net.Layers;
        net = trainNetwork(xTrain,yTrain,layers,options);
        save(net_path+"/net.mat",'net');  % Save the trained network
    
        fprintf("already training %d times \n",train_time);

    end

    for i = 1:test_num
        if i == 1
            save_amp_bias_txt = fopen(net_path+"/save_amp.txt",'w');
        else
            save_amp_bias_txt = fopen(net_path+"/save_amp.txt",'a');
        end
        fprintf(save_amp_bias_txt," amp = %f , bias = %f ,bandpower = %f \n" , save_amp(i), bias_save(i), band_power(i));
        if i == test_num
            fprintf(save_amp_bias_txt," load begin = %d , load end = %d  \n" , load_begin,load_end);
        end
        fclose(save_amp_bias_txt);
    end

    for i =1:test_num
        eval(['clear xTest',num2str(i)])
        eval(['clear yTest',num2str(i)])
    end
    clear xTrain yTrain
    pause(10)

end