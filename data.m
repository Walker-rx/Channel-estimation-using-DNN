clear
close all;

M = 4;
% filter_order = 1000;
% rp = 0.00057565;      
% rst = 1e-4;       % filter parameter used in function sam_rate_con
% 
% 
% origin_rate = 25e6;
% bw = origin_rate/2;
% ups_rate_transmit = 160e6;
data_num = 4000;
data_length = 1000;

t = datetime('now');
save_path_tmp = "data_save/"+t.Month+"."+t.Day;
% if(~exist(save_path,'dir'))
%     mkdir(char(save_path));
% end

K = 9; % channel length
h = randn(K,1); % channel
saveH = 'save_h';
eval([saveH,'=h;']);
save_path_h = "data_save/"+t.Month+"."+t.Day+"/data";
if(~exist(save_path_h,'dir'))
    mkdir(char(save_path_h));
end
save(save_path_h+"/save_h.mat",saveH);

%%
% ls_esi
%%
snr_begin = 2;
for snr = snr_begin:4:50
    save_path = save_path_tmp + "/data/snr"+snr;
    if(~exist(save_path,'dir'))
        mkdir(char(save_path));
    end
    snr_total = 0;
    for i = 1:data_num
        x = randi([0,M-1],[1,data_length]);
        %     x_mpam = real(pammod(x,M));
        y_tmp = conv(x,h);
        %     y = y_tmp((length(h)+1)/2:length(y_tmp)-(length(h)-1)/2);
        %     noise = 0 + 0.*randn(1,length(y_tmp));
        %     y = y_tmp + noise;
        y = awgn(y_tmp,snr,'measured');
        snr_loop = 10*log10(bandpower(y_tmp)/(bandpower(y)-bandpower(y_tmp)));
        snr_total = snr_total + snr_loop;
        snr_real = snr_total/i;     
        save_x = ['save_x_' num2str(i)];
        save_y = ['save_y_' num2str(i)];
        x = x.';
        y = y.';
        eval([save_x,'=x;']);
        eval([save_y,'=y;']);
        if i == 1
            save(save_path+"/save_x.mat",save_x);
            save(save_path+"/save_y.mat",save_y);
        else
            save(save_path+"/save_x.mat",save_x,'-append');
            save(save_path+"/save_y.mat",save_y,'-append');
        end
        if mod(i,40) == 0
            fprintf('loop = %d dB , snr = %d dB , save time=%d .\n',snr,round(snr_real),i);
        end
    end
    
    if snr == snr_begin
        fsnr = fopen(save_path_h+"/snr.txt",'w');
    else
        fsnr = fopen(save_path_h+"/snr.txt",'a');
    end
    fprintf(fsnr,' snr = %.8f \r\n',snr_real);
    fclose(fsnr);
end
