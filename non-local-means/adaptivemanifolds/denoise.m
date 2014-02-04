% Load file
s = load("Ammonit-Eo_u-cropped-141x152x180.vol.mat");

for [val key] = s
	data = val;
	name = key;
end

% Normalize image? 

% Save file
save("-mat-binary", "test.mat", "data");
