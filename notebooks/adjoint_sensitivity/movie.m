% =========================================================================
% This script is to making animation. 
% =========================================================================
%
picpath = '/Users/virrynay.wu/Documents/ECCO_Summer_School_2025/figures/'; 
img_path_list = dir(fullfile(picpath,'adjoint_sensitivity_tauu_lag*.png')); % or other image extension
img_num = length(img_path_list);
%%
% Create VideoWriter object
obj = VideoWriter([picpath, 'adjoint_sensitivity_tauu.mp4'], 'MPEG-4');
obj.FrameRate = 5; % Set frame rate

% Open the video file for writing
open(obj);

% Loop through all images
for j = 1:img_num
    % Get image filename
    image_name = fullfile(img_path_list(j).folder, img_path_list(j).name);
    
    % Read the image
    pdata = imread(image_name);
    
    % Write the frame to the video
    writeVideo(obj, pdata);
end

% Close the video file
close(obj);














