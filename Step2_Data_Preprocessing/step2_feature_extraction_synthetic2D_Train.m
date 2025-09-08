%% Synthetic Data Subvolume Extraction and Image Generation
% This script processes synthetic atomistic datasets, divides them into subvolumes, 
% and generates 2D projections (XY, XZ, YZ planes) for CNN training.

%% ============================= USER INPUTS =============================
baseFolder = '/home/tangy/cluster/Synthetic/';  % <-- Change this to your base folder
outputFolder = 'Train';   % <-- Folder to save generated images
numSyntheticFiles = 630;  % <-- Total number of synthetic datasets
cubeSize = 20;            % <-- Subvolume size in nm
gridSize = 100;           % <-- Grid size for projection images
minPointsThreshold = 300; % <-- Minimum number of points in subvolume to process
minClusterSize = 100;     % <-- Minimum cluster size to consider
resizeImageSize = [100, 100]; % <-- Final image size (pixels)
%% ======================================================================

iter = 1;  % Image counter
parameter = {}; 
groundtruth = [];

for syn = 1:numSyntheticFiles
    % Construct paths for input synthetic dataset
    folderName = ['GeneratedDatasetResults_Synthetic', num2str(syn)];
    fileName = ['Synthetic', num2str(syn), '_CuAtomsInClusterandSolidSolution.txt'];
    path = fullfile(baseFolder, folderName, fileName);

    % Read the dataset
    Data = readmatrix(path);

    % Compute number of cubes in x, y, z directions
    num_cubes = floor((max(Data) - min(Data)) / cubeSize);

    % Divide dataset into subvolumes
    data = cell(num_cubes);  % Preallocate
    for i = 1:num_cubes(1)
        for j = 1:num_cubes(2)
            for k = 1:num_cubes(3)
                x = min(Data(:, 1)) + cubeSize * (i - 1);
                y = min(Data(:, 2)) + cubeSize * (j - 1);
                z = min(Data(:, 3)) + cubeSize * (k - 1);

                log1 = (x <= Data(:, 1) & Data(:, 1) <= (x + cubeSize));
                log2 = (y <= Data(:, 2) & Data(:, 2) <= (y + cubeSize));
                log3 = (z <= Data(:, 3) & Data(:, 3) <= (z + cubeSize));
                data{i, j, k} = Data(log1 & log2 & log3, :);
            end
        end
    end

    % Process each subvolume
    for i = 1:size(data, 1)
        for j = 1:size(data, 2)
            for k = 1:size(data, 3)
                gt = data{i, j, k};

                % Remove invalid points (label = -1)
                gt(gt(:, 5) == -1, :) = [];

                if ~isempty(gt) && size(gt, 1) > minPointsThreshold
                    % Plot for visual check (optional)
                    % figure; scatter3(gt(:, 1), gt(:, 2), gt(:, 3), 5, '.')

                    % Count clusters
                    [uniqueVals, ~, idx] = unique(gt(:, 5));
                    counts = accumarray(idx, 1);
                    t = [uniqueVals, counts];
                    t_filtered = t(t(:, 2) >= minClusterSize, :);

                    cluster_num = size(t_filtered, 1);

                    % Scale coordinates for grid representation
                    minCoords = min(gt(:, 1:3));
                    maxCoords = max(gt(:, 1:3));
                    scaledCoords = (gt(:, 1:3) - minCoords) ./ (maxCoords - minCoords) * (gridSize - 1);

                    % Initialize 2D projections
                    Count_xy = zeros(gridSize, gridSize);
                    Count_xz = zeros(gridSize, gridSize);
                    Count_yz = zeros(gridSize, gridSize);

                    % Populate projections
                    for pt = 1:size(gt, 1)
                        ix = floor(scaledCoords(pt, 1)) + 1;
                        iy = floor(scaledCoords(pt, 2)) + 1;
                        iz = floor(scaledCoords(pt, 3)) + 1;
                        Count_xy(ix, iy) = Count_xy(ix, iy) + 1;
                        Count_xz(ix, iz) = Count_xz(ix, iz) + 1;
                        Count_yz(iy, iz) = Count_yz(iy, iz) + 1;
                    end

                    % Save projections as images
                    filename = fullfile(outputFolder, [num2str(iter), '.jpg']);
                    imwrite(count_rgb(Count_xy, resizeImageSize), filename);
                    parameter{iter} = [cluster_num];
                    groundtruth(iter, :) = [syn, i, j, k, cluster_num];
                    iter = iter + 1;

                    filename = fullfile(outputFolder, [num2str(iter), '.jpg']);
                    imwrite(count_rgb(Count_xz, resizeImageSize), filename);
                    parameter{iter} = [cluster_num];
                    groundtruth(iter, :) = [syn, i, j, k, cluster_num];
                    iter = iter + 1;

                    filename = fullfile(outputFolder, [num2str(iter), '.jpg']);
                    imwrite(count_rgb(Count_yz, resizeImageSize), filename);
                    parameter{iter} = [cluster_num];
                    groundtruth(iter, :) = [syn, i, j, k, cluster_num];
                    iter = iter + 1;
                end
            end
        end
    end
end

%% ============================= HELPER FUNCTION =============================
function img_resized = count_rgb(img, resizeImageSize)
    % Convert matrix to RGB image and resize
    figure; imagesc(img);
    clim([0, max(img(:))]);
    colormap_used = colormap;
    frame = getframe(gca);
    im_rgb = frame2im(frame);
    img_resized = imresize(im_rgb, resizeImageSize);
    close;
end
