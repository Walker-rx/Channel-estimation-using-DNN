%% Load data
save_amp = [];
band_power = [];
bias_save = [];
xTrain = [];
yTrain = [];
bias_loop = cell(1,2);

amp_loop_data(1) = data{data_loop}(1,1);
bias_loop_data{1} = data{data_loop}(1,2:end);
amp_loop_data(2) = data{data_loop}(2,1);
bias_loop_data{2} = data{data_loop}(2,2:end);
test_num = 0;
for i = 1:length(amp_loop_data)
    amp_folder = amp_loop_data(i);

    for j = 1:length(bias_loop_data{i})
        bias_folder = bias_loop_data{i}(j);
        if bias_folder ~=0
            test_num = test_num + 1;
            load_path = save_path + "/data/10M/amp"+amp_folder+"/bias"+bias_folder+"/mat";
            fprintf(" %d looptimes , %d training times , load amp = %f , bias = %d \n",train_loop_time,train_time,amp_folder,bias_folder);
            load_data
            totalNum = data_num*split_num;
            trainNum = floor(totalNum*train_percent);
            xTrain_tmp = x(1:trainNum);
            yTrain_tmp = y(1:trainNum);
            xTest_tmp = x(trainNum+1:end);
            yTest_tmp = y(trainNum+1:end);

            amp_loop = 32000*amp_folder;
            save_amp = [ save_amp 10*log10(amp_loop^2) ];
            bias_save = [ bias_save bias_folder];
            xTrain_tmp = cellfun(@(cell1)(cell1*amp_loop),xTrain_tmp,'UniformOutput',false);
            xTest_tmp = cellfun(@(cell1)(cell1*amp_loop),xTest_tmp,'UniformOutput',false);

            xTrain = [xTrain xTrain_tmp];
            yTrain = [yTrain yTrain_tmp];

            xTest_name = ['xTest',num2str(test_num)];
            yTest_name = ['yTest',num2str(test_num)];
            eval([xTest_name,'=xTest_tmp;']);
            eval([yTest_name,'=yTest_tmp;']);

            clear x y
        end       
    end
end

%%  Normalize data
load_norm_path = "data_save/light_data_3.10/data/10M/rand_bias0.3/";
norm_mat = load(load_norm_path+"/save_norm.mat");
norm_names = fieldnames(norm_mat);
norm_factor = gather(eval(strcat('norm_mat.',norm_names{1})));

xTrain = cellfun(@(cell1)(cell1*norm_factor),xTrain,'UniformOutput',false);

for i = 1:test_num
    band_power = [band_power bandpower(xTrain{10+(i-1)*trainNum})];
end

for i = 1:test_num
    xTest_nor = eval(['xTest',num2str(i)]);
    xTest_nor = cellfun(@(cell1)(cell1*norm_factor),xTest_nor,'UniformOutput',false);
    eval([['xTest',num2str(i)],'= xTest_nor;']);
end

%%  Reshape data
bias_loop_data{1}(find(bias_loop_data{1}==0)) = [];
bias_loop_data{2}(find(bias_loop_data{2}==0)) = [];
bias_all = [ bias_loop_data{1} , bias_loop_data{2}];
for i = 1:numel(xTrain)
    xTrain{i} = toeplitz(xTrain{i}(h_order:-1:1),xTrain{i}(h_order:end));
    xTrain{i} = [xTrain{i}; bias_all(floor((i-1)/trainNum)+1)*ones(1,size(xTrain{i},2) )];
    yTrain{i} = reshape(yTrain{i}(1:split_length*rate_times),outputSize,split_length);
    yTrain{i} = yTrain{i}(:,1:size(xTrain{i},2));
end

for i = 1:test_num
    xtop_tem = eval(['xTest',num2str(i)]);
    ytop_tem = eval(['yTest',num2str(i)]);
    for j = 1:numel(xtop_tem)
        xtop_tem{j} = toeplitz(xtop_tem{j}(h_order:-1:1),xtop_tem{j}(h_order:end));
        xtop_tem{j} = [xtop_tem{j}; bias_all(i)*ones(1,size(xtop_tem{j},2) )];
        ytop_tem{j} = reshape(ytop_tem{j}(1:split_length*rate_times),outputSize,split_length);
        ytop_tem{j} = ytop_tem{j}(:,1:size(xtop_tem{j},2));
    end
    eval([['xTest',num2str(i)],'= xtop_tem;']);
    eval([['yTest',num2str(i)],'= ytop_tem;']);
end

%%
