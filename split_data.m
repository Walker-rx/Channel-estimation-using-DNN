function [data] = split_data(amp_scope,bias_scope)
    amp_order = randperm(length(amp_scope));
    amp_data = amp_scope(amp_order);
    data = cell(1,length(amp_data));
    for i = 1:length(amp_data)
        bias_order = randperm(length(bias_scope));
        bias_data = bias_scope(bias_order);
        data{i} = [amp_data(i) bias_data];
    end
end