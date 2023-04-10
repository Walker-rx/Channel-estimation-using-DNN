function [bias_data,bias_scope_tmp_after] = split_bias(bias_scope_tmp)
    bias_data = cell(1,2);
    if  length(bias_scope_tmp) >=13 || length(bias_scope_tmp) <=21
        bias_order = randperm(length(bias_scope_tmp),floor(length(bias_scope_tmp)/2));
        bias_data{1} = bias_scope_tmp(bias_order);
        bias_scope_tmp(bias_order) = [];
        bias_data{2} = bias_scope_tmp;
        bias_scope_tmp = [];
    else
        bias_order = randperm(length(bias_scope_tmp),10);
        bias_data{1} = bias_scope_tmp(bias_order);
        bias_scope_tmp(bias_order) = [];
        bias_order = randperm(length(bias_scope_tmp),10);
        bias_data{2} = bias_scope_tmp(bias_order);
        bias_scope_tmp(bias_order) = [];
    end
    
    if length(bias_data{1})-length(bias_data{2})>0
        bias_data{2} = [bias_data{2} zeros(1,length(bias_data{1})-length(bias_data{2}))];
    elseif length(bias_data{1})-length(bias_data{2})<0
        bias_data{1} = [bias_data{1} zeros(1,length(bias_data{2})-length(bias_data{1}))];
    end
    bias_scope_tmp_after = bias_scope_tmp;
end