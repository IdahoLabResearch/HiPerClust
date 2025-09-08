function [CrPoints, crUsed] = generateCrClusters(Centers, Radii, ClusterSizeInfo, CrPercentage, TotalPoints)
TotalCr = round(CrPercentage * TotalPoints);
CrPoints = [];
crUsed = 0;
maxPerCluster = floor(TotalCr / size(Centers,1));

for i = 1:size(Centers,1)
    rx = Radii(i,1); ry = Radii(i,2); rz = Radii(i,3);
    numPoints = maxPerCluster;

    x = randn(numPoints,1) * rx;
    y = randn(numPoints,1) * ry;
    z = randn(numPoints,1) * rz;

    pts = [x + Centers(i,1), y + Centers(i,2), z + Centers(i,3)];
    pts(:,4)=i;
    CrPoints = [CrPoints; pts];
    crUsed = crUsed + numPoints;
end
end