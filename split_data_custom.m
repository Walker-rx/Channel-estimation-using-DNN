function [data,bias_scope_after] = split_data_custom(amp_data,bias_scope)

    if length(bias_scope)>=12
        data = cell(1,2);
        [bias_data,bias_scope_after] = split_bias(bias_scope);
        if length(amp_data)>1
            data{1} = [amp_data(1) bias_data{1};...
                amp_data(2) bias_data{2}];
            data{2} = [amp_data(1) bias_data{2};...
                amp_data(2) bias_data{1}];
        else
            data{1} = [amp_data(1) bias_data{1};...
                amp_data(1) bias_data{2}];
        end

    else

        data = cell(1,1);
        [bias_data,bias_scope_after] = split_bias(bias_scope);

        if length(amp_data)>1            
            data{1} = [amp_data(1) bias_data{1};...
                amp_data(2) bias_data{2}];
        else
            data{1} = [amp_data(1) bias_data{1}];
        end
    end
 
end