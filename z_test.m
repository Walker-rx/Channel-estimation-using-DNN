del_num = 1;

new_fin_point=cell(1,300-del_num);
new_ori=cell(1,300-del_num);
new_rec=cell(1,300-del_num);
new_ups=cell(1,300-del_num);

amp=19;
load_path = "data_save/light_data_3.11/data/10M/rand_bias0.3/amp"+amp+"/mat";

x_mat = load(load_path+"/save_signal_ori.mat");
y_mat = load(load_path+"/save_signal_received_real_send1.mat");
fin_mat = load(load_path+"/save_fin_syn_point_real_send1.mat");
upsample_norm_mat = load(load_path+"/save_upsample_norm.mat");
x_names = fieldnames(x_mat);
y_names = fieldnames(y_mat);
fin_names = fieldnames(fin_mat);
upsample_norm_names = fieldnames(upsample_norm_mat);

for name_order = del_num+1:300 
    signal_ori = gather(eval(strcat('x_mat.',x_names{name_order})));
    signal_received = gather(eval(strcat('y_mat.',y_names{name_order})));
    fin_syn_point = gather(eval(strcat('fin_mat.',fin_names{name_order})));
    upsample_norm = gather(eval(strcat('upsample_norm_mat.',upsample_norm_names{name_order})));

    new_fin_point{name_order-del_num}=fin_syn_point;
    new_ori{name_order-del_num}=signal_ori;
    new_rec{name_order-del_num}=signal_received;
    new_ups{name_order-del_num}=upsample_norm;
end

save_path_mat = "data_save/new_mat/amp"+amp+"/mat";
if(~exist(save_path_mat,'dir'))
    mkdir(char(save_path_mat));
end

for i = 1:300-del_num
    save_fin_name = ['save_fin_syn_point_real_send1_' num2str(i)];
    save_ori_name = ['save_signal_ori_' num2str(i)];
    save_rec_name = ['save_signal_received_real_send1_' num2str(i)];
    save_ups_name = ['save_upsample_' num2str(i)];

    eval([save_fin_name,'=new_fin_point{i};']); 
    eval([save_ori_name,'=new_ori{i};']); 
    eval([save_rec_name,'=new_rec{i};']); 
    eval([save_ups_name,'=new_ups{i};']); 
    if i == 1
        save(save_path_mat+"/save_fin_syn_point_real_send1.mat",save_fin_name);
        save(save_path_mat+"/save_signal_ori.mat",save_ori_name);    
        save(save_path_mat+"/save_signal_received_real_send1.mat",save_rec_name);
        save(save_path_mat+"/save_upsample_norm.mat",save_ups_name);
    else
        save(save_path_mat+"/save_fin_syn_point_real_send1.mat",save_fin_name,'-append');
        save(save_path_mat+"/save_signal_ori.mat",save_ori_name,'-append');    
        save(save_path_mat+"/save_signal_received_real_send1.mat",save_rec_name,'-append');
        save(save_path_mat+"/save_upsample_norm.mat",save_ups_name,'-append');
    end
end