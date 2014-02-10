function output_volume = double2vol(input_volume, min_value, max_value)
    vol_range = max_value - min_value;
    output_volume = min_value + vol_range * input_volume;
end
