function data = Generate_2002UnfavorableData(shapeNumber)
    
    figure;
    % Parabola
    if shapeNumber == 1
       X=[-5:0.01:5];
       Y = X.^2 + normrnd(0, 1);
       Z = X.^2 + 20 + normrnd(0, 1);    
       scatter(X,Y);
       hold on;
       scatter(X,Z);
    
    % Two half circle
    elseif shapeNumber == 2
         X=[-5:0.01:5];
%         X = -5 + (5+5)*rand(1001,1);
        Y = sqrt(25-X.^2); 
        Z = sqrt(36-X.^2);
        scatter(X,Y);
        hold on;
        scatter(X,Z);

    elseif shapeNumber == 3
        r0 = 1 ;  % inner radius
        r1 = 3. ;   % outer radius
        % circles
        th = linspace(0,2*pi, 1000) ;
        x0 = r0*sin(th) ; 
        y0 = r0*cos(th) ;
        
        x1 = r1*sin(th) ; 
        y1 = r1*cos(th) ;
        
        scatter(x0,y0);
        hold on
        scatter(x1,y1) ;
        
        data = zeros(2000,2);
        data(1:1000, 1) = x0;
        data(1:1000, 2) = y0;
        data(1001:2000, 1) = x1;
        data(1001:2000, 2) = y1;
    end
    
    if shapeNumber == 1 || shapeNumber == 2
        data = zeros(2002, 2);
        data(1:1001,1) = X';
        data(1002:2002, 1) = X';
        data(1:1001,2) = Y';
        data(1002:2002, 2) = Z';
    end

end
