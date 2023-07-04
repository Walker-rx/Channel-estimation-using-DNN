clear
close all
% layers = [
%     imageInputLayer([28 28 1],'Name','input')
%     
%     convolution2dLayer(5,16,'Padding','same','Name','conv_1')
%     batchNormalizationLayer('Name','BN_1')
%     reluLayer('Name','relu_1')
%     
%     convolution2dLayer(3,32,'Padding','same','Stride',2,'Name','conv_2')
%     batchNormalizationLayer('Name','BN_2')
%     reluLayer('Name','relu_2')
%     convolution2dLayer(3,32,'Padding','same','Name','conv_3')
%     batchNormalizationLayer('Name','BN_3')
%     reluLayer('Name','relu_3')
%     
%     additionLayer(2,'Name','add')
%     
%     averagePooling2dLayer(2,'Stride',2,'Name','avpool')
%     fullyConnectedLayer(10,'Name','fc')
%     softmaxLayer('Name','softmax')
%     classificationLayer('Name','classOutput')];
% lgraph = layerGraph(layers);
% 
% skipConv = convolution2dLayer(1,32,'Stride',2,'Name','skipConv');
% lgraph = addLayers(lgraph,skipConv);
% lgraph = connectLayers(lgraph,'relu_1','skipConv');
% lgraph = connectLayers(lgraph,'skipConv','add/in2');
% figure
% plot(lgraph);

snr_5=[5.772437	8.631669	15.262309	19.314387	22.059148	24.548482	29.261113	35.386892	37.694872	38.76328];
snr_25=[3.569465	6.394834	12.865074	16.997767	19.881711	22.432959	27.179607	34.216753	36.950297	38.460093];
snr_45 = [1.676935	4.834928	11.314311	15.229514	18.419383	20.697199	25.667021	32.620154	35.460106	36.92877];
snr_65 = [0.686499	3.598261	10.085992	14.129606	17.206815	19.475128	24.521041	31.555595	34.332083	35.766314];
snr_73 = [0.218393	3.150669	9.699638	13.79098	16.648389	19.271752	24.198605	31.126211	34.01312	35.396533];

nmse_5 = [-8.069465	-10.303716	-15.339474	-18.073015	-19.789446	-20.395456	-21.447775	-21.902035	-21.284674	-19.664808];
nmse_25 = [-6.062476	-8.14185	-13.023319	-16.524574	-17.989582	-18.986422	-20.635044	-22.23106	-22.171865	-21.197046];
nmse_45 = [-4.877201	-6.779188	-11.930966	-15.148913	-17.140079	-18.161438	-19.844313	-21.623846	-21.267969	-21.915041];
nmse_65 = [-4.034187	-5.818727	-10.901716	-14.015452	-16.11978	-18.12504	-20.342161	-20.766941	-22.144562	-25.32];
nmse_73 = [-3.795619	-5.418969	-10.497126	-13.845817	-15.99898	-17.875875	-21.183714	-23.29291	-23.794937	-24.24688];

ls_5 = [-6.622677	-8.837613	-14.369054	-17.300668	-19.329919	-18.894821	-20.100611	-16.526612	-14.786227	-14.078735];
ls_25 = [-4.889008	-6.875563	-12.556705	-15.706051	-17.342244	-19.346557	-20.426616	-20.548579	-21.474436	-18.840816];
ls_45 = [-3.837864	-5.645632	-10.970421	-14.316199	-16.426549	-18.260197	-19.59633	-22.791849	-18.72356	-22.306892];
ls_65 = [-3.174772	-4.825302	-9.996099	-13.417286	-15.749176	-16.787494	-18.311584	-25.746236	-24.033269	-17.785358];
ls_73 = [-2.881637	-4.478927	-9.386319	-13.291792	-15.186611	-16.758478	-20.16917	-23.939922	-23.582948	-20.731104];
plot(snr_5,nmse_5,'k+-', 'Linewidth', 2, 'MarkerSize', 8)
hold on
plot(snr_25,nmse_25,'ro-', 'Linewidth', 2, 'MarkerSize', 8)
plot(snr_45,nmse_45,'bs-', 'Linewidth', 2, 'MarkerSize', 8)
plot(snr_65,nmse_65,'g^-', 'Linewidth', 2, 'MarkerSize', 8)
plot(snr_73,nmse_73,'m*-', 'Linewidth', 2, 'MarkerSize', 8)
legend('bias = 50mA','bias = 250mA','bias = 450mA','bias = 650mA','bias = 730mA');
xlabel("SNR")
ylabel("NMSE")
ylim([-26 -1.5])
title("不同偏置电流时的信道拟合性能")

