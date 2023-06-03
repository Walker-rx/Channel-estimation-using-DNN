%% Load data
if data_loop == 1
    save_amp = [];
    band_power = [];
    bias_save = []; 
    bias_all = [];
end
xTrain_loop = [];
yTrain_loop = [];

test_num = 0;
data_tmp = data{data_loop};
for row_loop = 1:size(data_tmp,1)        
    amp_folder = data_tmp(row_loop,1);
    bias_loop_data = data_tmp(row_loop,2:end);
    bias_loop_data(find(bias_loop_data==0)) = [];
    bias_all = [bias_all bias_loop_data];
    for bias_loop = 1:length(bias_loop_data)
        bias_folder = bias_loop_data(bias_loop);
        test_num = test_num + 1;
        load_path = load_path_ini + "/data/10M/amp"+amp_folder+"/bias"+bias_folder+"/mat";
        fprintf(" %d looptimes , %d training times , load amp = %f , bias = %d , \n load begin = %d , load end = %d \n",...
            train_loop_time,train_time,amp_folder,bias_folder,load_begin,load_end);
        load_data
        totalNum = data_num*split_num;
        trainNum = floor(totalNum*train_percent);
        xTrain_tmp = x(1:trainNum);
        yTrain_tmp = y(1:trainNum);
        xTest_tmp = x(trainNum+1:end);
        yTest_tmp = y(trainNum+1:end);

        amp_loop = 32000*amp_folder;
        save_amp = [ save_amp amp_loop ];
        bias_save = [ bias_save bias_folder];
        xTrain_tmp = cellfun(@(cell1)(cell1*amp_loop),xTrain_tmp,'UniformOutput',false);
        xTest_tmp = cellfun(@(cell1)(cell1*amp_loop),xTest_tmp,'UniformOutput',false);

        xTrain_loop = [xTrain_loop xTrain_tmp];
        yTrain_loop = [yTrain_loop yTrain_tmp];

        xTest_name = ['xTest',num2str(test_num)];
        yTest_name = ['yTest',num2str(test_num)];
        eval([xTest_name,'=xTest_tmp;']);
        eval([yTest_name,'=yTest_tmp;']);

        clear x y
    end
end

%%  Normalize data
test_num = round(test_num);
load_norm_path = "data_save/light_data_3.10/data/10M/rand_bias0.3/";
norm_mat = load(load_norm_path+"/save_norm.mat");
norm_names = fieldnames(norm_mat);
norm_factor = gather(eval(strcat('norm_mat.',norm_names{1})));

xTrain_loop = cellfun(@(cell1)(cell1*norm_factor),xTrain_loop,'UniformOutput',false);

for i = 1:test_num
    band_power = [band_power bandpower(xTrain_loop{10+(i-1)*trainNum})];
end

for i = 1:test_num
    xTest_nor = eval(['xTest',num2str(i)]);
    xTest_nor = cellfun(@(cell1)(cell1*norm_factor),xTest_nor,'UniformOutput',false);
    eval([['xTest',num2str(i)],'= xTest_nor;']);
end

%%  Reshape data
for i = 1:numel(xTrain_loop)
    xTrain_loop{i} = toeplitz(xTrain_loop{i}(h_order:-1:1),xTrain_loop{i}(h_order:end));
    xTrain_loop{i} = [xTrain_loop{i}; single( bias_loop_data(floor((i-1)/trainNum)+1) )*ones(1,size(xTrain_loop{i},2) )];
    yTrain_loop{i} = reshape(yTrain_loop{i}(1:split_length*rate_times),outputSize,split_length);
    yTrain_loop{i} = yTrain_loop{i}(:,1:size(xTrain_loop{i},2));
end

for i = 1:test_num
    xtop_tem = eval(['xTest',num2str(i)]);
    ytop_tem = eval(['yTest',num2str(i)]);
    for j = 1:numel(xtop_tem)
        xtop_tem{j} = toeplitz(xtop_tem{j}(h_order:-1:1),xtop_tem{j}(h_order:end));
        xtop_tem{j} = [xtop_tem{j}; single( bias_loop_data(i) )*ones(1,size(xtop_tem{j},2) )];
        ytop_tem{j} = reshape(ytop_tem{j}(1:split_length*rate_times),outputSize,split_length);
        ytop_tem{j} = ytop_tem{j}(:,1:size(xtop_tem{j},2));
    end
    eval([['xTest',num2str(i)],'= xtop_tem;']);
    eval([['yTest',num2str(i)],'= ytop_tem;']);
end


% xValidation = xTest1;
% yValidation = yTest1;
for i = 1:test_num
    xValidation_tem = eval(['xTest',num2str(i)]);
    yValidation_tem = eval(['yTest',num2str(i)]);
    xValidation = [xValidation,xValidation_tem{1},xValidation_tem{2}];
    yValidation = [yValidation,yValidation_tem{1},yValidation_tem{2}];
end
xTrain = [xTrain xTrain_loop];
yTrain = [yTrain yTrain_loop];
% xTrain = xTrain_loop;
% yTrain = yTrain_loop;
%%
