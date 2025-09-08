% Generate random cluster centers within a 3D volume, ensuring minimum spacing
function [centers, minDistance] = generateClusterCenters(xMin, xMax, yMin, yMax, zMin, zMax, numClusters, sepFactor, maxRadius, clusterInfo)

% Apply scaling to shrink boundaries
xMin = xMin * 0.8; xMax = xMax * 0.8;
yMin = yMin * 0.8; yMax = yMax * 0.8;
zMin = zMin * 0.8; zMax = zMax * 0.8;

% Number of random points to test
candidateCount = 1e6;

% Define valid region considering max cluster radius
xLower = xMin + maxRadius;
xUpper = xMax - maxRadius;
yLower = yMin + maxRadius;
yUpper = yMax - maxRadius;
zLower = zMin + maxRadius;
zUpper = zMax - maxRadius;

% Generate random candidate points in the reduced region
randX = (xUpper - xLower) .* rand(candidateCount, 1) + xLower;
randY = (yUpper - yLower) .* rand(candidateCount, 1) + yLower;
randZ = (zUpper - zLower) .* rand(candidateCount, 1) + zLower;

% Initialize with the first point
selectedX = randX(1);
selectedY = randY(1);
selectedZ = randZ(1);

% Minimum distance allowed between any two centers
minAllowedDist = sepFactor * maxRadius;
count = 2;

% Loop through candidates and select those meeting spacing criteria
for i = 2:candidateCount
    dx = selectedX - randX(i);
    dy = selectedY - randY(i);
    dz = selectedZ - randZ(i);
    distances = sqrt(dx.^2 + dy.^2 + dz.^2);

    if min(distances) >= minAllowedDist
        selectedX(count, 1) = randX(i);
        selectedY(count, 1) = randY(i);
        selectedZ(count, 1) = randZ(i);
        count = count + 1;
    end
end

% Check if we have enough clusters
if size(selectedX, 1) < numClusters
    error('Not enough space for requested clusters. Increase candidate count or adjust separation.');
elseif size(selectedX, 1) > numClusters
    idx = randperm(size(selectedX, 1), numClusters);
    selectedX = selectedX(idx);
    selectedY = selectedY(idx);
    selectedZ = selectedZ(idx);
end

% Combine into matrix
centers = [selectedX, selectedY, selectedZ];

% Compute minimum pairwise distance
distPairs = pdist(centers);
minDistance = min(distPairs);

end