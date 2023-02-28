clear
pilot_length = 2047;
zero_length = 3000;
save_path = "data_save/light_data_2.28";
savePath_txt = save_path + "/result/25M/8pam/optimal_nmse";
if(~exist(savePath_txt,'dir'))
    mkdir(char(savePath_txt));
end
amp_begin = -4;
amp_end = 50;
for amp = amp_begin:2:amp_end
    fprintf("amp = %d \n",amp);
    load_path = save_path + "/data/25M/8pam/amp"+amp+"/mat";
    y_mat = load(load_path+"/save_signal_received_real_send1.mat");
    fin_mat = load(load_path+"/save_fin_syn_point_real_send1.mat");

    y_names = fieldnames(y_mat);
    fin_names = fieldnames(fin_mat);
    data_num = numel(y_names);

    noise = cell(1,data_num);
    data= cell(1,data_num);

    for name_order = 1:data_num
        signal_received = gather(eval(strcat('y_mat.',y_names{name_order})));
        fin_syn_point = gather(eval(strcat('fin_mat.',fin_names{name_order})));
        noise{name_order} = signal_received(fin_syn_point+pilot_length*6+500:fin_syn_point+pilot_length*6+500+1000-1);
        data{name_order} = signal_received(fin_syn_point+(pilot_length+zero_length)*6+1500:fin_syn_point+(pilot_length+zero_length)*6+1500+1000-1);
    end
    noise = gather(noise);
    data = gather(data);
    clear y_mat fin_mat y_names fin_names
    nmse = cellfun(@(no,da)10*log10(sum(no.^2)/sum(da.^2)),noise,data);
    Nmse = mean(nmse);
    if amp == amp_begin
        save_amp = fopen(savePath_txt+"/save_amp.txt",'w');
        save_Nmse = fopen(savePath_txt+"/optimal_nmse.txt",'w');
    else
        save_amp = fopen(savePath_txt+"/save_amp.txt",'a');
        save_Nmse = fopen(savePath_txt+"/optimal_nmse.txt",'a');
    end
    fprintf(save_amp,'%d \r\n',amp);
    fprintf(save_Nmse,'%f \r\n',Nmse);
    fclose(save_amp);
    fclose(save_Nmse);
end
