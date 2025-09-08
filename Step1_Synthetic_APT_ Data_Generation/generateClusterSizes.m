% Create a list of cluster dimensions for irregular clusters based on input specifications
function [clusterDims, numClusters, maxRadius] = generateClusterSizes(clusterInfo)

clusterDims = [];

% Iterate through each row in the input configuration
for idx = 1:size(clusterInfo, 1)
    count = clusterInfo(idx, 1); % Number of clusters for this set
    baseX = clusterInfo(idx, 2); % Mean X radius
    deltaX = clusterInfo(idx, 3); % Fluctuation in X radius
    baseY = clusterInfo(idx, 4); % Mean Y radius
    deltaY = clusterInfo(idx, 5); % Fluctuation in Y radius
    baseZ = clusterInfo(idx, 6); % Mean Z radius
    deltaZ = clusterInfo(idx, 7); % Fluctuation in Z radius

    % Generate random sizes with uniform variation
    sizeX = baseX + (rand(count, 1) * 2 * deltaX) - deltaX;
    sizeY = baseY + (rand(count, 1) * 2 * deltaY) - deltaY;
    sizeZ = baseZ + (rand(count, 1) * 2 * deltaZ) - deltaZ;

    % Combine into a single matrix for this group
    clusterDims = [clusterDims; [sizeX, sizeY, sizeZ]];
end

% Calculate maximum dimension across all clusters
maxRadius = max(clusterDims(:));
numClusters = size(clusterDims, 1);

end
