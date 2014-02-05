#!/usr/bin/octave -qf

% Check args
if (nargin < 2)
    usage('%s', [program_name() ' input_file' ' output_file']);
end

% Read args
input_file = argv(){1};
output_file = argv(){2};

% Load file
input_file_data = load(input_file);

for [val key] = input_file_data
    input_volume = val;
    input_name = key;
end

% Convert volume to scale [0 1]
[input_volume min_value max_value] = vol2double(input_volume);



% Save file
output_volume = input_volume;
save('-mat-binary', output_file, 'output_volume');
