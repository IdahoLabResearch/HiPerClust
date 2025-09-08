%% Synthetic Fe-Cr Data Generator
% This script generates synthetic Fe-Cr datasets with clustered Cr regions.
% Suitable for Atom Probe Tomography (APT) simulation and machine learning model training.

%% ==================== USER INPUT SECTION ====================
Dim = [80, 80, 80];          % Volume dimensions [X, Y, Z] in nm  <-- USER INPUT
TotalNumPoints = 7000000;  % Total number of atoms (Fe + Cr)    <-- USER INPUT
CrPercentage = 0.12;         % Chromium atomic fraction (e.g., 12%) <-- USER INPUT

Density = 9.5e23;            % Atom density in atoms/m^3          <-- USER INPUT
DensityError = 0.2e23;       % Density uncertainty (optional)

NumClusters = 100;           % Number of Cr-rich clusters         <-- USER INPUT
clusterInfo = [NumClusters, ...
                   1.5, 0.2, ... % X radius + fluctuation (nm)
                   1.5, 0.2, ... % Y radius + fluctuation (nm)
                   1.5, 0.1];    % Z radius + fluctuation (nm)
MinClusterSeparation = 5;    % Minimum separation between clusters (nm) <-- USER INPUT

OutputFolder = './SyntheticData'; % Folder to save generated data <-- USER INPUT
if ~exist(OutputFolder, 'dir')
    mkdir(OutputFolder);
end

% Derived values
TotalNumCr = round(CrPercentage * TotalNumPoints);
TotalNumFe = TotalNumPoints - TotalNumCr;

%% ==================== DATA GENERATION LOOP ====================
for syn = 1:40  % <-- USER INPUT: Range of dataset indices to generate
    % Output file name
    NameOfTheOutputFile = fullfile(OutputFolder, sprintf('Synthetic_%d.mat', syn)); % <-- USER INPUT: Name of the output file

    fprintf('Generating dataset %d...\n', syn);

    %% Step 1: Generate cluster sizes
    % r: radii of clusters, dmax: max cluster size, TotalNumberOfClusters: actual number
    [r, TotalNumberOfClusters, dmax] = generateClusterSizes(clusterInfo);

    %% Step 2: Find cluster centers within the volume
    [ClusterCenters, MinDist] = generateClsuterCenters(0, Dim(1), ...
                                                  0, Dim(2), ...
                                                  0, Dim(3), ...
                                                  NumClusters, ...
                                                  MinClusterSeparation, ...
                                                  dmax, ...
                                                  clusterInfo);

    %% Step 3: Generate clustered Cr atoms
    [CrPoints_clustered, CrUsed] = generateCrClusters(ClusterCenters, r, ...
                                                      clusterInfo, ...
                                                      CrPercentage, ...
                                                      TotalNumPoints);

    %% Step 4: Generate background Cr and Fe atoms
    RemainingCr = TotalNumCr - CrUsed;

    % Hard-coded formula for Fe background atoms:
    % Adjust this logic based on your use case!
    RemainingFe = (1500000 / 20); % <-- USER INPUT or REVISE formula

    [CrPoints_bg, FePoints] = generateBackground(Dim, ClusterCenters, dmax, ...
                                                RemainingCr, RemainingFe);

    %% Combine all points and labels
    AllPoints = [CrPoints_clustered(:,1:3); CrPoints_bg; FePoints];
    Labels = [CrPoints_clustered(:,4); zeros(size(CrPoints_bg,1),1); zeros(size(FePoints,1),1)];

    Data = [AllPoints, Labels];

    %% Save the generated dataset
    save(NameOfTheOutputFile, 'Data');

    %% Optional Visualization (Uncomment to view)
    % figure;
    % scatter3(AllPoints(:,1), AllPoints(:,2), AllPoints(:,3), 5, Labels, 'filled');
    % axis equal;
    % title('Fe (blue) and Cr (red)');
    % xlabel('X (nm)'); ylabel('Y (nm)'); zlabel('Z (nm)');
end

fprintf('Synthetic dataset generation completed. Files saved in: %s\n', OutputFolder);
