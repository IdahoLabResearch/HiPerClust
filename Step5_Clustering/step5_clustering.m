%% ================================
%  K-Means + HDBSCAN Clustering Script
%  Description:
%    - Loads raw 3D coordinates and model predictions
%    - Performs preliminary clustering with K-Means
%    - Refines clustering using HDBSCAN with parameter tuning
% =================================

clear; clc;
%% ================================
% 1. USER INPUTS%
=================================
% Paths to required files
rawDataFile      = '12Cr.mat';                 % Raw 3D coordinates file <-- USER INPUT
predictionFile   = 'ConvNeXtLarge_12.mat';     % Predictions file from the step3_model_training (can also use 'Resnet50_12.mat') <-- USER INPUT

% Clustering parameters
threshold        = 2;                          % Distance threshold for assigning clusters <-- USER INPUT
minPersistence   = 0.06;                       % Minimum persistence threshold for HDBSCAN filtering <-- USER INPUT

% Python environment & HDBSCAN script
pythonEnvPath = '/home/tangy/miniforge3/envs/APT/bin/python'; % Path to Python executable <-- USER INPUT
hdbscanScript = 'hdbscanImanSecond_9.py'; % Python HDBSCAN script name <-- USER INPUT

% Temporary files for HDBSCAN input/output
tempDataFile     = 'TempHDBSCANfile_9.txt';
tempParamFile    = 'TempHDBSCANparameters_9.txt';
% =================================

%% ================================
%  2. LOAD DATA
% =================================
fprintf('Loading data...\n');
load(rawDataFile);           % Raw 3D coordinates (stored in variable "data") 
load(predictionFile);        % Model predictions (stored in variable "predictions") 

% Convert predictions to integer clusters (incremented by 1)
predictions = round(predictions + 1);

%% ================================
% 3. PRELIMINARY CLUSTERING (K-Means)
% =================================
fprintf('Running K-Means with %d clusters...\n', pred_n);
[idx, C] = kmeans(gt(:, 1:3), pred_n);% Compute distances to cluster centers, gt is the Raw 3D coordinates loaded from rawDataFile, pred_n is the model preidicitions from predictionFile
D = pdist2(gt(:, 1:3), C);
[minDist, idx] = min(D, [], 2);

% Assign clusters, mark outliers as -1
clusterLabels = idx;
clusterLabels(minDist > threshold) = -1;

% Count points in valid clusters
counts = accumarray(clusterLabels(clusterLabels > 0), 1);
label = clusterLabels;
fprintf('Preliminary clustering complete. Found %d clusters.\n', length(counts));

%% ================================
% 4. DETERMINE HDBSCAN PARAMETERS
% =================================
m = min(counts); % Minimum cluster size from K-means
n = round(0.1 * m); % Min samples (10% of m)
fprintf('HDBSCAN parameters: min_cluster_size = %d, min_samples = %d\n', m, n);

%% ================================
% 5. PREPARE DATA FOR HDBSCAN
% =================================
writematrix(gt(:, 1:3), tempDataFile, 'Delimiter', ' '); % Save coordinates
fid = fopen(tempParamFile, 'wt');
fprintf(fid, '%d\n', m); % Write min_cluster_size
fprintf(fid, '%d', n); % Write min_samples
fclose(fid);

%% ================================
% 6. RUN HDBSCAN (Python)
% =================================
fprintf('Running HDBSCAN via Python script...\n');
system(sprintf('%s %s', pythonEnvPath, hdbscanScript));

%% ================================
% 7. POST-PROCESS HDBSCAN RESULTS
% =================================
hdbscanCluster_CNN = processHDBSCANResults(tempDataFile);

% Filter clusters based on persistence and size
hdbscanCluster_CNN([hdbscanCluster_CNN.persistence] < minPersistence | [hdbscanCluster_CNN.clustersize] < m) = [];
fprintf('Final HDBSCAN clustering complete. Retained %d clusters.\n', length(hdbscanCluster_CNN));


%% ================================
% 8. POST-PROCESS Function
% =================================
function clusters = processHDBSCANResults(dataFile)
% processHDBSCANResults
% Reads HDBSCAN output files (Labels, Persistence, Probabilities)
% and organizes cluster information including atom positions.
%
% INPUT:
% dataFile - name of the file containing original coordinates (e.g., 'TempHDBSCANfile.txt')
%
% OUTPUT:
% clusters - structure array with fields:
% labels - cluster label (integer)
% probabilities - membership probabilities for each point
% persistence - persistence value for the cluster
% atomPositions - coordinates of points in the cluster
% clustersize - number of points in the cluster
%


%% Load HDBSCAN output files
labels = importdata('Labels.txt'); % Cluster labels (-1 = noise)
persistence = importdata('Persistence.txt'); % Persistence score per cluster
probabilities = importdata('Probabilities.txt'); % Membership probabilities
dataPoints = importdata(dataFile); % Original coordinates



%% Handle empty clustering result
if isempty(persistence)
    clusters = {};
    return;
end

%% Identify unique labels
noiseLabel = -1;
uniqueLabels = unique(labels);

% Check if noise is present
if ismember(noiseLabel, uniqueLabels)
    validLabels = setdiff(uniqueLabels, noiseLabel);
else
    validLabels = uniqueLabels;
end

%% Build cluster structure
clusters = struct('labels', {}, 'probabilities', {}, ...
    'persistence', {}, 'atomPositions', {}, ...
    'clustersize', {});

for i = 1:numel(validLabels)
    currentLabel = validLabels(i);
    
    % Get all rows corresponding to the current cluster
    rows = find(labels == currentLabel);
    
    % Fill cluster structure
    clusters(i).labels = currentLabel;
    clusters(i).probabilities = probabilities(rows);
    clusters(i).persistence = persistence(i);
    clusters(i).atomPositions = dataPoints(rows, 1:3);
    clusters(i).clustersize = size(clusters(i).atomPositions, 1);
end
end