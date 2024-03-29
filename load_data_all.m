
pilot_length = 2047;
zero_length = 3000;
x_mat = load(load_path+"/save_signal_ori.mat");
y_mat = load(load_path+"/save_signal_received_real_send1.mat");
fin_mat = load(load_path+"/save_fin_syn_point_real_send1.mat");
upsample_norm_mat = load(load_path+"/save_upsample_norm.mat");
x_names = fieldnames(x_mat);
y_names = fieldnames(y_mat);
fin_names = fieldnames(fin_mat);
upsample_norm_names = fieldnames(upsample_norm_mat);
data_num = numel(x_names);

x = cell(1,data_num*split_num);
y = cell(1,data_num*split_num);
upsample_norm = zeros(1,data_num);
data_length = 10000;
split_length = data_length/split_num;
% x = cell(data_num,1);
% y = cell(data_num,1);
for name_order = 1:data_num  
    signal_ori = gather(eval(strcat('x_mat.',x_names{name_order})));
    signal_received = gather(eval(strcat('y_mat.',y_names{name_order})));
    fin_syn_point = gather(eval(strcat('fin_mat.',fin_names{name_order})));
    upsample_norm(name_order) = gather(eval(strcat('upsample_norm_mat.',upsample_norm_names{name_order})));
    data_ori = signal_ori(pilot_length+zero_length+1:end);
    data_received = signal_received(fin_syn_point + (pilot_length+zero_length)*rate_times : end);
    for i = 1:split_num
        x{split_num*(name_order-1)+i} = [zeros(1,15),data_ori(split_length*(i-1)+1 : split_length*i)]/upsample_norm(name_order);
        y{split_num*(name_order-1)+i} = data_received(split_length*rate_times*(i-1)+1 : split_length*rate_times*i);
    end
end
x = gather(x);
y = gather(y);
clear x_mat y_mat fin_mat x_names y_names fin_names upsample_norm_mat upsample_norm_names


