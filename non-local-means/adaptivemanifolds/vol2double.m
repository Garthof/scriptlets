function [output_volume min_value max_value] = vol2double(input_volume)
    min_value = min(min(min(input_volume)));
    max_value = max(max(max(input_volume)));

    output_volume = double(input_volume-min_value);
    output_volume = output_volume / double(max_value-min_value);
end
