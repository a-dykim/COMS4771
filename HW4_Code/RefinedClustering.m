function [c_result, X_labels] = RefinedClustering(in_data, in_r, in_K, in_bVisualize)

[n, dim] = size(in_data);

% Step1: Construct W fr om the input data (in_data)
distanceMatrix = zeros(n,n);
r_nearest_matrix = zeros(n,n);

for i = 1:n
    for j = 1:n
        distanceMatrix(i,j) = norm(in_data(i,:) - in_data(j,:));
    end
    for r_index = 1:(in_r+1)
        [minVal, minPos] = min(distanceMatrix(i,:));
        if (minPos ~= i)
            r_nearest_matrix(i, minPos) = 1;
        end
        distanceMatrix(i, minPos) = realmax;
    end
end

W = r_nearest_matrix;
for i = 1:n
    for j = 1:n
       W(i,j) = (W(j,i) == 1 || r_nearest_matrix(i,j) == 1) ;
    end
end

% Step 2: Construct a diagonal matrix D as follows:
D = zeros(n,n);


for i = 1:n
    D(i,i) = sum(W(i,:));
end

% Step 3: Define L = D - W
L = D - W;

%
[eigenvectors, eigenvalues] = eig(L);

[d, ind] = sort(diag(eigenvalues));
sortedEigenVectorMatrix = eigenvectors(:,ind);

V = sortedEigenVectorMatrix(:,1:in_K);

X_labels = zeros(n,1);

% Coloring data point accroding to the final labels
if (in_bVisualize == 1 && dim == 2 && in_K <= 8)
    colors = ['c','g','b','w','k','y', 'm', 'r'];
    figure;

    for k_index = 1:in_K
        X_labels(V(:,k_index) ~= 0, 1) = k_index;
        scatter(in_data(V(:,k_index) ~= 0,1), in_data(V(:,k_index) ~= 0, 2), colors(k_index));
        c_result(k_index,1) = mean(in_data(V(:,k_index) ~= 0, 1));
        c_result(k_index,2) = mean(in_data(V(:,k_index) ~= 0, 2));

        hold on;
    end

    plot(c_result(:,1), c_result(:,2),'r*');
end

end