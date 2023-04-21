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
data_type = 3;
if data_type == 3
    bias_begin = 0.05;
    bias_step = 0.04;
    bias_end = 0.85;

    amp = 1;
    amp_loop_begin = amp;
    amp_loop_end = amp;
    amp_loop_step = 1;
    amp_loop_num = (amp_loop_end - amp_loop_begin)/amp_loop_step + 1 ;


    amp_begin = amp;
    amp_norm = 0;

    load_begin = 1;
    load_end = 200;
    data_num = load_end-load_begin+1;
elseif data_type == 2
    bias_begin = 0.05;
    bias_step = 0.04;
    bias_end = 0.85;

    amp = 0.1613;
    amp_loop_begin = amp;
    amp_loop_end = amp;
    amp_loop_step = 1;
    amp_loop_num = (amp_loop_end - amp_loop_begin)/amp_loop_step + 1 ;


    amp_begin = amp;
    amp_norm = 0;

    load_begin = 1;
    load_end = 100;
    data_num = load_end-load_begin+1;
end

looptime = 0;
ver = amp;

folder_name = "light_data_"+folder+", v"+ver;
disp(folder_name);

%%
for loop = amp_loop_begin:amp_loop_step:amp_loop_end
    save_amp = zeros(1,amp_loop_num);
    for bias = bias_begin : bias_step :bias_end
        %% Load data
        looptime = looptime + 1;
        load_path = save_path + "/data/10M/amp"+amp+"/bias"+bias+"/mat";
        load_data
        %% Normalize data
        %     x = cellfun(@(cell1)(cell1*100*1.1^amp),x,'UniformOutput',false);
        amp_loop = 32000*(amp_begin+(loop-1)*amp_norm);
        x = cellfun(@(cell1)(cell1*amp_loop),x,'UniformOutput',false);
        save_amp(looptime) = 10*log10(amp_loop^2);

        load_path = "data_save/light_data_3.10/data/10M/rand_bias0.3/";
        norm_mat = load(load_path+"/save_norm.mat");
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
        savePath_result = save_path + "/result"+data_type+"/"+t.Month+"."+t.Day+"/norm_LS_"+ver;
        if(~exist(savePath_result,'dir'))
            mkdir(char(savePath_result));
        end

        if bias == bias_begin
            save_Nmse = fopen(savePath_result+"/save_Nmse.txt",'w');
            save_bandpower = fopen(savePath_result+"/save_bandpower.txt",'w');
            save_amp_txt = fopen(savePath_result+"/save_amp.txt",'w');
            %             save(savePath_result+"/save_h.mat",saveH);
        else
            save_Nmse = fopen(savePath_result+"/save_Nmse.txt",'a');
            save_bandpower = fopen(savePath_result+"/save_bandpower.txt",'a');
            save_amp_txt = fopen(savePath_result+"/save_amp.txt",'a');
            %             save(savePath_result+"/save_h.mat",saveH,'-append');
        end
        fprintf(save_Nmse,'%f \r\n',Nmse);
        fprintf(save_bandpower,'%f \r\n',band_power);
        fprintf(save_amp_txt,"%f \n" , save_amp(looptime));
        fclose(save_Nmse);
        fclose(save_bandpower);
        fclose(save_amp_txt);
        fprintf(' bias = %d , nmse = %.6g \r\n',bias,Nmse);

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
fprintf(save_parameter," LS \r\n amp begin = %.5f , amp end = %.5f , amp step = %.5f \r\n ",...
    amp_loop_begin, amp_loop_end, amp_loop_step);
fprintf(save_parameter,"bias begin = %.2f , bias end = %.2f , bias step = %.2f \r\n ",...
    bias_begin, bias_end, bias_step);
fprintf(save_parameter,"data_num = %d , split num = %d , train num = %d\r\n",data_num,split_num,trainNum);
fprintf(save_parameter," origin rate = %e , receive rate = %e \n",ori_rate,rec_rate);
fprintf(save_parameter," H order = %d ,related num = %d \n",h_order,related_num);
fprintf(save_parameter," Add zero num = %d \n",add_zero);
fclose(save_parameter);
