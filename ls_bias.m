clear 
close all

t = datetime('now');
folder = '4.14';
save_path = "data_save/light_data_"+folder;
ori_rate = 10e6;
rec_rate = 60e6;
rate_times = rec_rate/ori_rate;
related_num = 8;
h_order = related_num*rate_times;
add_zero = round(h_order/2);
split_num = 1;

%% Loop parameter settings

bias_scope = 0.05:0.04:0.85;
bias_begin = bias_scope(1);
amp_scope = [0.005 0.007 0.015 0.024 0.034 0.045 0.08 0.18 0.25 0.3];
amp_loop_num = length(amp_scope) ;

load_begin = 1;
load_end = 60;
data_num = load_end-load_begin+1;

looptime = 0;

%%
for loop = 1:amp_loop_num
    amp = amp_scope(loop);
    ver = amp;
    folder_name = "light_data_"+folder+", v"+ver;
    disp(folder_name);

    save_amp = zeros(1,amp_loop_num);
    for bias_loop = 1:length(bias_scope)
        bias = bias_scope(bias_loop);
        %% Load data
        looptime = looptime + 1;
        load_path = "/home/xliangseu/ruoxu/equalization-using-DNN/data_save/light_data_5.21/data/10M/amp"+amp+"/bias"+bias+"/mat";
        load_data
        %% Normalize data
        amp_loop = 32000*amp;
        x = cellfun(@(cell1)(cell1*amp_loop),x,'UniformOutput',false);
        save_amp(looptime) = 10*log10(amp_loop^2);

        load_path_norm = "data_save/light_data_3.10/data/10M/rand_bias0.3/";
        norm_mat = load(load_path_norm+"/save_norm.mat");
        norm_names = fieldnames(norm_mat);
        norm_factor = gather(eval(strcat('norm_mat.',norm_names{1})));
        x = cellfun(@(cell1)(cell1*norm_factor),x,'UniformOutput',false);
        %%

        totalNum = numel(x);
        trainNum = floor(totalNum*0.90);
        xTrain = x(1:trainNum);
        yTrain = y(1:trainNum);
        xTest = x(trainNum+1:end);
        yTest = y(trainNum+1:end);

        band_power = bandpower(xTrain{1});

        h = zeros(h_order,rate_times);

        %% Reshape data
        for i = 1:numel(xTrain)
            xTrain{i} = toeplitz(xTrain{i}(h_order:-1:1),xTrain{i}(h_order:end)).';
        end

        for i = 1:rate_times
            yTrain2 = cell(1,numel(yTrain));
            for j = 1:numel(yTrain)
                yTrain2{j} = yTrain{j}(i:rate_times :i+rate_times *(size(xTrain{1},1)-1));
            end

            h_tmp = zeros(size(xTrain{1},2),size(xTrain{1},2));
            for j = 1:numel(xTrain)
                h_tmp = h_tmp + xTrain{j}.'*xTrain{j};
            end

            h_tmp2 = zeros(size(xTrain{1},2) , size(xTrain{1},1)*numel(xTrain));
            for j = 1:numel(xTrain)
                h_tmp2(: , size(xTrain{1},1)*(j-1)+1 : size(xTrain{1},1)*j) = h_tmp\xTrain{j}.';
            end

            h_hat = zeros(size(xTrain{1},2) , 1);
            for j = 1:numel(xTrain)
                h_hat = h_hat + h_tmp2(:,size(xTrain{1},1)*(j-1)+1:size(xTrain{1},1)*j)*yTrain2{j}.';
            end

            h(:,i) = h_hat;
        end        
        fprintf(" LS matrix generation completed \n");

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
%         savePath_result = save_path + "/result1"+"/"+t.Month+"."+t.Day+"/norm_LS/amp"+ver;
        savePath_result = save_path + "/result1/norm_LS/amp"+ver;
        if(~exist(savePath_result,'dir'))
            mkdir(char(savePath_result));
        end

        if bias == bias_begin
            save_Nmse = fopen(savePath_result+"/save_Nmse.txt",'w');
            save_bandpower = fopen(savePath_result+"/save_bandpower.txt",'w');
            save_amp_txt = fopen(savePath_result+"/save_amp.txt",'w');
            save_bias_txt = fopen(savePath_result+"/save_bias.txt",'w');
            %             save(savePath_result+"/save_h.mat",saveH);
        else
            save_Nmse = fopen(savePath_result+"/save_Nmse.txt",'a');
            save_bandpower = fopen(savePath_result+"/save_bandpower.txt",'a');
            save_amp_txt = fopen(savePath_result+"/save_amp.txt",'a');
            save_bias_txt = fopen(savePath_result+"/save_bias.txt",'a');
            %             save(savePath_result+"/save_h.mat",saveH,'-append');
        end
        fprintf(save_Nmse,'%f \r\n',Nmse);
        fprintf(save_bandpower,'%f \r\n',band_power);
        fprintf(save_amp_txt,"%f \n" , save_amp(looptime));
        fprintf(save_bias_txt,"%f \n" , bias);
        fclose(save_Nmse);
        fclose(save_bandpower);
        fclose(save_amp_txt);
        fclose(save_bias_txt);
        fprintf('amp= %f , bias = %d , nmse = %.6g \r\n',amp,bias,Nmse);

    end
%         %%  Save data
%         savePath_result = save_path + "/result"+data_type+"/"+t.Month+"."+t.Day+"/10M/LS/norm_LS"+ver;
%         if(~exist(savePath_result,'dir'))
%             mkdir(char(savePath_result));
%         end
% 
% %         saveH = ['save_h_' num2str(looptime)];
% %         eval([saveH,'=h;']);
%         if bias == bias_begin
%             save_Nmse = fopen(savePath_result+"/save_Nmse.txt",'w');
%             save_bandpower = fopen(savePath_result+"/save_bandpower.txt",'w');
%             save_amp_txt = fopen(savePath_result+"/save_amp.txt",'w');
% %             save(savePath_result+"/save_h.mat",saveH);
%         else
%             save_Nmse = fopen(savePath_result+"/save_Nmse.txt",'a');
%             save_bandpower = fopen(savePath_result+"/save_bandpower.txt",'a');
%             save_amp_txt = fopen(savePath_result+"/save_amp.txt",'a');
% %             save(savePath_result+"/save_h.mat",saveH,'-append');
%         end
%         fprintf(save_Nmse,'%f \r\n',Nmse);
%         fprintf(save_bandpower,'%f \r\n',band_power);
%         fprintf(save_amp_txt,"%f \n" , save_amp(looptime));
%         fclose(save_Nmse);
%         fclose(save_bandpower);
%         fclose(save_amp_txt);
%         fprintf(' bias = %d , nmse = %.6g \r\n',bias,Nmse);
    
end
save_parameter = fopen(savePath_result+"/save_parameter.txt",'w');
fprintf(save_parameter,"\n \n");
fprintf(save_parameter," LS \r\n ");

fprintf(save_parameter,"amp =");
for i = 1:amp_loop_num
    fprintf(save_parameter," %f,",amp_scope(i));
end
fprintf(save_parameter,"\r\n");

fprintf(save_parameter," bias =");
for i = 1:length(bias_scope)
    fprintf(save_parameter," %f,",bias_scope(i));
end
fprintf(save_parameter,"\r\n");

fprintf(save_parameter,"data_num = %d , split num = %d , train num = %d\r\n",data_num,split_num,trainNum);
fprintf(save_parameter," origin rate = %e , receive rate = %e \n",ori_rate,rec_rate);
fprintf(save_parameter," H order = %d ,related num = %d \n",h_order,related_num);
fprintf(save_parameter," Add zero num = %d \n",add_zero);
fclose(save_parameter);
