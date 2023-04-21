clear
close all

t = datetime('now');
folder = '4.14';
save_path = "data_save/light_data_"+folder;

ver = 50;

savePath_txt = save_path + "/result1/"+t.Month+"."+t.Day+"/mix_bias_amp/Threenonlinear"+ver;   
savePath_mat = save_path + "/result1/"+t.Month+"."+t.Day+"/mix_bias_amp/Threenonlinear"+ver; 
if(~exist(savePath_txt,'dir'))
    mkdir(char(savePath_txt));
end
if(~exist(savePath_mat,'dir'))
    mkdir(char(savePath_mat));
end

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
LearnRateDropPeriod = 8;
LearnRateDropFactor = 0.1;
inilearningRate = 1e-2;


%%
fprintf("This is Threenonlinear network , ini learningRate = %e  , DropPeriod = %d , DropFactor = %f \n",...
    inilearningRate,LearnRateDropPeriod,LearnRateDropFactor);
fprintf("Hidden Units = %d , v%d \n",numHiddenUnits,ver)

train_loop_time = 1;
% bias_scope = 0.05:0.04:0.85;
% amp_scope_ini = [0.1613 0.32106 0.48082 0.64058 0.8003 1];
bias_scope = 0.85;
amp_scope_ini = 1;
data_scope = {[1 80] [81 160] [161 200] [201 280] [281 300]};
% data_scope = {[1 100]};
train_percent = 0.95;
total_data_num = 0;

dataset_order = cell(1,length(amp_scope_ini));
amp_data = [];
bias_data = cell(1,2);
train_time = 0;
amp_scope = amp_scope_ini;
while ~isempty(amp_scope)
    if length(amp_scope)>1
        amp_order = randperm(length(amp_scope),2);
        amp_data(1) = amp_scope(amp_order(1));
        amp_data(2) = amp_scope(amp_order(2));
        amp_scope(amp_order) = [];

        bias_scope_tmp = bias_scope;
        while ~isempty(bias_scope_tmp)
            miniBatchSize = 399;
            [data,bias_scope_tmp] = split_data_custom(amp_data,bias_scope_tmp);         

%             amp_loop_data = [];
%             bias_loop_data = cell(1,size(data{1},1));
            for load_scope = 1:numel(data_scope)
                load_begin = data_scope{load_scope}(1);
                load_end = data_scope{load_scope}(2);
                data_num = load_end-load_begin+1;
                total_data_num = total_data_num + data_num;
                for data_loop = 1:numel(data)
                    train_time = train_time+1;
                    load_bias_amp_custom
                    %% Initialize network
                    validationFrequency = floor(numel(xTrain)/miniBatchSize/2);
                    dnn_option

                    %% Train network                  
                    dnn_train_default(train_time, savePath_mat, xTrain, yTrain, layers, options, test_num,...
                                                   save_amp, bias_save, band_power, load_begin, load_end);

                end           % for data_loop = 1:numel(data) ; amp_scope size > 1
            end               % for load_scope = 1:numel(data_scope) ; amp_scope size > 1
        end                   % while ~isempty(bias_scope_tmp) ; amp_scope size > 1
        
    else

        amp_data(1) = amp_scope(1);
        amp_scope(1) = [];
        bias_scope_tmp = bias_scope;
        while ~isempty(bias_scope_tmp)
            [data,bias_scope_tmp] = split_data_custom(amp_data,bias_scope_tmp);         

%             amp_loop_data = [];
%             bias_loop_data = cell(1,size(data{1},1));
            for load_scope = 1:numel(data_scope)
                load_begin = data_scope{load_scope}(1);
                load_end = data_scope{load_scope}(2);
                data_num = load_end-load_begin+1;
                total_data_num = total_data_num + data_num;
                for data_loop = 1:numel(data)
                    train_time = train_time+1;
                    load_bias_amp_custom
                    %% Initialize network
                    miniBatchSize = floor(numel(xTrain)/40);
                    validationFrequency = floor(numel(xTrain)/miniBatchSize/2);
                    dnn_option

                    %% Train network                  
                    dnn_train_default(train_time, savePath_mat, xTrain, yTrain, layers, options, test_num,...
                                                    save_amp, bias_save, band_power, load_begin, load_end);

                end           % for data_loop = 1:numel(data) ; amp_scope size = 1
            end               % for load_scope = 1:numel(data_scope) ; amp_scope size = 1
        end                   % while ~isempty(bias_scope_tmp) ; amp_scope size = 1     

    end              % if length(amp_scope)>1
    
end      % while ~isempty(amp_scope)

%% Save data
save_parameter = fopen(savePath_txt+"/save_parameter.txt",'w');
fprintf(save_parameter,"\n \n");
fprintf(save_parameter," Default training \n");
fprintf(save_parameter," Threenonlinear ,\r\n 40 iteration per epoch , \r\n ");
fprintf(save_parameter,"ini learningRate = %e ,\r\n DropPeriod = %d , DropFactor = %f ,\r\n ",inilearningRate,LearnRateDropPeriod, LearnRateDropFactor);
fprintf(save_parameter,"amp =");
for i = 1:length(amp_scope_ini)
    fprintf(save_parameter," %f,",amp_scope_ini(i));
end
fprintf(save_parameter,"\r\n");
fprintf(save_parameter," bias =");
for i = 1:length(bias_scope)
    fprintf(save_parameter," %f,",bias_scope(i));
end
fprintf(save_parameter,"\r\n");
fprintf(save_parameter," data num = %d , split num = %d , train num = %d\r\n",total_data_num,split_num,total_data_num*split_num*train_percent);
fprintf(save_parameter," validationFrequency is floor(numel(xTrain)/miniBatchSize/2) \n");
fprintf(save_parameter," origin rate = %e , receive rate = %e \n",ori_rate,rec_rate);
fprintf(save_parameter," H order = %d ,related num = %d \n",h_order,related_num);
fprintf(save_parameter," Hidden Units = %d \n",numHiddenUnits);
fprintf(save_parameter," Add zero num = %d \n",add_zero);
fclose(save_parameter);


fprintf("\n Training end ..." + ...
    "\n Threenonlinear , ini learningRate = %e ,  DropPeriod = %d , DropFactor = %f , data_num = %d \n",...
    inilearningRate,  LearnRateDropPeriod, LearnRateDropFactor, total_data_num);
fprintf(" result saved in %s \n",savePath_mat);


    
