#!/usr/bin/octave -qf

% Check args
if (nargin < 2)
    usage('%s', [program_name() ' input_file' ' output_file']);
end

% Read args
input_file = argv(){1};
output_file = argv(){2};

% Load file
s = load(input_file);

for [val key] = s
	data = val;
	name = key;
end

% Normalize image?

% Save file
save('-mat-binary', output_file, 'data');
