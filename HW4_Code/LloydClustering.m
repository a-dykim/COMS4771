function [c_result, X_labels] = LloydClustering(in_data, K, bVisualize)
    [n, d] = size(in_data);
    
    % X is a (n x d) matrix, n is # of data, d is dimensionality of each
    % data
    X = in_data;
    X_labels = zeros(n,1);
   
    % c_result is a (K x d) matrix
    c_result = zeros(K, d);
    
    % randomly assign starting points to each c_result(i), i = 1,...,K
    for k_index=1:K
        c_result(k_index, :) = normrnd(0, 2, [1,d]);
    end
    
    bVeryTinyChange = 0;
    iterationLimit = 1;
    
    while bVeryTinyChange == 0 && (iterationLimit <= 1000)
        old_c_result = c_result;
        
        % Assignment of data (Exapectation)
        for i = 1:n
            discrepancy = c_result - X(i,:);
            [minValue, minIndex] = min(diag(discrepancy * discrepancy'));        
            X_labels(i) = minIndex;
        end

        % Find the optimal centers given data partition
        for k_index = 1:K
            foundResult = find(X_labels == k_index);
            foundCount = size(foundResult);
            
            if foundCount > 0
                c_result(k_index,:) = mean(X(foundResult, :));
            else
                c_result(k_index,:) = c_result(k_index, :);
            end            
        end 
        
        bVeryTinyChange = all(diag((old_c_result - c_result) * (old_c_result - c_result)') <= 0.1);
        iterationLimit = iterationLimit + 1;
    end
    
    % Coloring data point accroding to the final labels
    if (bVisualize == 1 && d == 2 && K <= 8)
        colors = ['y','m','c','r','g','b','w','k'];
        figure;
        
        for k_index = 1:K
            scatter(X(X_labels == k_index,1), X(X_labels == k_index,2), colors(k_index));
            hold on;
        end

        plot(c_result(:,1), c_result(:,2),'r*');
    end
end
