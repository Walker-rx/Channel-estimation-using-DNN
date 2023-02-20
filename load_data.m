x_mat = load(load_path+"/save_x.mat");
y_mat = load(load_path+"/save_y.mat");
x_names = fieldnames(x_mat);
y_names = fieldnames(y_mat);
data_num = min(length(x_names),length(y_names));

x = cell(1,data_num);
y = cell(1,data_num);
% x = cell(data_num,1);
% y = cell(data_num,1);
for name_order = 1:data_num  
    x{name_order} = eval(strcat('x_mat.',x_names{name_order}));
    y{name_order} = eval(strcat('y_mat.',y_names{name_order}));
end

clear x_mat y_mat x_names y_names
% clearvars -except x y data_num save_path snr snr_begin snr_end