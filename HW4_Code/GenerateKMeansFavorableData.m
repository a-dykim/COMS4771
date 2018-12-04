function X = GenerateKMeansFavorableData(dim, K, nTotalPoints)

    sizeOfEachCluster = round(nTotalPoints / K);
    X = zeros(nTotalPoints, dim);
    
    mu_list = zeros(1, K);
    for k_index = 1:K
        mu_list(k_index) = 5 * (k_index - 1);
    end
    
    for k_index = 1:K
       startIndex = (sizeOfEachCluster * (k_index-1) + 1);
       endIndex = startIndex + sizeOfEachCluster - 1;
       
       for i = 0:(sizeOfEachCluster-1)
            X(startIndex + i, :) = normrnd(mu_list(k_index), 0.1, [1, dim]);
       end
    end
end