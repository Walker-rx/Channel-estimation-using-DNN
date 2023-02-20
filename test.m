clear
close all
% a = cell(1,1000);
% b = cell(1,1000);
% for i = 1:5
%     a{i} = i:i+5;
%     b{i} = i:i+5;
% end
% c = cellfun(@(cell1,cell2)isequal(cell1,cell2),a,b);
% d = cellfun(@(cell1,cell2)mse(cell1,cell2),a,b);
% acc = mean(c);
% mse = mean(d);
% clear a b
for i = 1:5
progress = ['snr = ',num2str(i),', save progress is ',num2str(i*100),'%'];
disp(progress);
end

