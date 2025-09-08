function [CrPoints_bg, FePoints] = generateBackground(Dim, Centers, dmax, NumCr, NumFe)
total = NumCr + NumFe;
pts = rand(total,3) .* Dim;

% remove points too close to clusters
keep = true(total,1);
for i = 1:size(Centers,1)
    d = sqrt(sum((pts - Centers(i,:)).^2,2));
    keep = keep & (d > dmax);
end
pts = pts(keep,:);

if size(pts,1) < total
    warning('Not enough background space. Reducing background atoms.');
    total = size(pts,1);
    NumCr = round((NumCr / (NumCr + NumFe)) * total);
    NumFe = total - NumCr;
end

CrPoints_bg = pts(1:NumCr,:);
FePoints = pts(NumCr+1:NumCr+NumFe,:);
end