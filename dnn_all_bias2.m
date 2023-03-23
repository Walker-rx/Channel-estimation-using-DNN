
%% Loop parameter settings
data_type = 2;
if data_type == 1
    bias_begin = 0.1;
    bias_step = 0.05;
    bias_end = 0.8;
    bias_loop_num = (bias_end-bias_begin)/bias_step+1;

    amp_loop_begin = 1;
    amp_loop_end = 1;
    amp_loop_step = 1;
    amp_loop_num = (amp_loop_end - amp_loop_begin)/amp_loop_step + 1 ;


    amp_begin = 1;
    amp_norm = 0;
elseif data_type == 2
    bias_begin = 0.05;
    bias_step = 0.04;
    bias_end = 0.85;
    bias_loop_num = (bias_end-bias_begin)/bias_step+1;

    amp_loop_begin = 1;
    amp_loop_end = 1;
    amp_loop_step = 1;
    amp_loop_num = (amp_loop_end - amp_loop_begin)/amp_loop_step + 1 ;


    amp_begin = 0.1613;
    amp_norm = 0;
end

%%
fprintf("This is Threenonlinear network , ini learningRate = %e , min batch size = %d , DropPeriod = %d , DropFactor = %f \n",...
    inilearningRate,miniBatchSize,LearnRateDropPeriod,LearnRateDropFactor);
fprintf("Hidden Units = %d , v%d \n",numHiddenUnits,ver)

%% Load data
test_num2 = 0;
bias_loop = 0;
save_amp2 = zeros(1,round(bias_loop_num*amp_loop_num));
band_power2 = zeros(1,round(bias_loop_num*amp_loop_num));
for bias = bias_begin : bias_step :bias_end
    
    bias_loop = bias_loop + 1;
    test_num_amp = 0;

    for loop = amp_loop_begin : amp_loop_step :amp_loop_end
        test_num_amp = test_num_amp + 1;
        load_path = save_path + "/data"+data_type+"/10M/bias"+bias+"/amp"+loop+"/mat";
        fprintf(" bias = %f , load amp = %d \n",bias,loop);
        load_data
        totalNum = data_num*split_num;
        trainNum2 = floor(totalNum*0.95);
        xTrain_tmp = x(1:trainNum2);
        yTrain_tmp = y(1:trainNum2);
        xTest_amp_tmp = x(trainNum2+1:end);
        yTest_amp_tmp = y(trainNum2+1:end);

        %     xTrain_tmp = cellfun(@(cell1)(cell1*100*1.1^amp),xTrain_tmp,'UniformOutput',false);
        %     xTest_tmp = cellfun(@(cell1)(cell1*100*1.1^amp),xTest_tmp,'UniformOutput',false);

        amp_loop = 32000*(amp_begin+(loop-1)*amp_norm);
        save_amp2((bias_loop-1)*amp_loop_num+test_num_amp) = 10*log10(amp_loop^2);
        xTrain_tmp = cellfun(@(cell1)(cell1*amp_loop),xTrain_tmp,'UniformOutput',false);
        xTest_amp_tmp = cellfun(@(cell1)(cell1*amp_loop),xTest_amp_tmp,'UniformOutput',false);

        xTest_amp_name = ['xTest_amp_',num2str(test_num_amp)];
        yTest_amp_name = ['yTest_amp_',num2str(test_num_amp)];
        eval([xTest_amp_name,'=xTest_amp_tmp;']);
        eval([yTest_amp_name,'=yTest_amp_tmp;']);

        if loop == amp_loop_begin
            xTrain_amp = xTrain_tmp;
            yTrain_amp = yTrain_tmp;
        else
            xTrain_amp = [xTrain_amp xTrain_tmp];
            yTrain_amp = [yTrain_amp yTrain_tmp];
        end
        clear x y
    end

    %%  Normalize data
    totaltrain = numel(xTrain_amp);
    % norm_cell = xTrain{floor(totaltrain/2)};
    % norm_factor = 1/norm(norm_cell)*sqrt(length(norm_cell));
    % load_path = "data_save/light_data_2.28/result/3.1/25M/8pam/mix_amp/Twononlinear";

    load_path = "data_save/light_data_3.10/data/10M/rand_bias0.3/";
    norm_mat = load(load_path+"/save_norm.mat");
    norm_names = fieldnames(norm_mat);
    norm_factor = gather(eval(strcat('norm_mat.',norm_names{1})));

    xTrain_amp = cellfun(@(cell1)(cell1*norm_factor),xTrain_amp,'UniformOutput',false);

    for i = 1:amp_loop_num
        band_power2((bias_loop-1)*amp_loop_num+i) = bandpower(xTrain_amp{10+(i-1)*trainNum2});
    end

    for i = 1:test_num_amp
        xTest_nor = eval(['xTest_amp_',num2str(i)]);
        xTest_nor = cellfun(@(cell1)(cell1*norm_factor),xTest_nor,'UniformOutput',false);
        eval([['xTest_amp_',num2str(i)],'= xTest_nor;']);
    end

    %%  Reshape data
    for i = 1:numel(xTrain_amp)
        xTrain_amp{i} = toeplitz(xTrain_amp{i}(h_order:-1:1),xTrain_amp{i}(h_order:end));
        xTrain_amp{i} = [xTrain_amp{i}; bias*ones(1,size(xTrain_amp{i},2) )];
        yTrain_amp{i} = reshape(yTrain_amp{i}(1:split_length*rate_times),outputSize,split_length);
        yTrain_amp{i} = yTrain_amp{i}(:,1:size(xTrain_amp{i},2));
    end
    for i = 1:test_num_amp
        xtop_tem = eval(['xTest_amp_',num2str(i)]);
        ytop_tem = eval(['yTest_amp_',num2str(i)]);
        for j = 1:numel(xtop_tem)
            xtop_tem{j} = toeplitz(xtop_tem{j}(h_order:-1:1),xtop_tem{j}(h_order:end));
            xtop_tem{j} = [xtop_tem{j}; bias*ones(1,size(xtop_tem{j},2) )];
            ytop_tem{j} = reshape(ytop_tem{j}(1:split_length*rate_times),outputSize,split_length);
            ytop_tem{j} = ytop_tem{j}(:,1:size(xtop_tem{j},2));
            %             xTest{i} = toeplitz(xTest{i}(inputSize:-1:1),xTest{i}(inputSize:end));
            %             yTest{i} = reshape(yTest{i}(1:6000),outputSize,1000);
            %             yTest{i} = yTest{i}(:,1:size(xTest1{i},2));
        end
        eval([['xTest_amp_',num2str(i)],'= xtop_tem;']);
        eval([['yTest_amp_',num2str(i)],'= ytop_tem;']);
    end

    %%
    test_num2 = test_num2 + test_num_amp;
%     if bias == bias_begin
%         xTrain = xTrain_amp;
%         yTrain = yTrain_amp;
%     else
        xTrain = [xTrain  xTrain_amp];
        yTrain = [yTrain  yTrain_amp];
%     end

    for i = 1:test_num_amp
        xTest_name = ['xTest',num2str(test_num+(bias_loop-1)*test_num_amp+i)];
        yTest_name = ['yTest',num2str(test_num+(bias_loop-1)*test_num_amp+i)];
        xTest_bias_tmp = eval(['xTest_amp_',num2str(i)]);
        yTest_bias_tmp = eval(['yTest_amp_',num2str(i)]);
        eval([xTest_name,'=xTest_bias_tmp;']);
        eval([yTest_name,'=yTest_bias_tmp ;']);
    end
    
end
