clear;  % 清空工作區，刪除所有變數
clc;    % 清空命令窗口
warning('off');  % 關閉警告
rng(123);  % 設置隨機數生成器的種子，確保結果可重現

%% 讀取需要預測數據
fileName = 'step1_output_PredictionData.csv'; % csv 檔路徑，這是需要讀取的資料檔案
data = readtable(fileName); % 讀取 CSV 檔案並將其存儲為表格

% 提取第一列資料，這可能是關於某個特定變數或識別碼
firstColumn = data{:, 1};

% 去除重複值並對數據進行排序，將唯一的值提取出來
uniqueList = unique(firstColumn); % unique 函數去除重複項並返回排序過的結果
numFiles = length(uniqueList); % 計算唯一值的個數，這是文件的數量或樣本數

% 定義獎牌類別，這裡有三個獎牌類別和一個 "All" 類別
typelist = ["Gold", "Silver", "Bronze", "All"];

% 初始化一個空的表格，用於存儲最終結果
resultsTable = table();

% 初始化一個空表格，用於存儲每個類別的信賴區間
confidenceIntervals = table();

% 初始化一個空表格，用於存儲每個類別的權重參數
weightsTable = table();

% 設置信賴水準為 0.05，這意味著 95% 信賴區間
alpha = 0.05;  % 95% 信賴區間

% 計算總的任務數量，這是唯一的文件數量乘以每個類別的數量
totalTasks = numFiles * length(typelist);

% 初始化當前任務計數器
currentTask = 0;

%% 迴圈處理每個檔
for fileIdx = 1:numFiles  % 迴圈處理每個唯一文件（每個文件代表一個運動項目）

    for i = 1:length(typelist)  % 迴圈處理每種獎牌類型（例如金、銀、銅或綜合）

        rng(123);  % 設置隨機數生成器的種子，確保結果可重現

        currentTask = currentTask + 1; % 更新任務計數器，表示當前正在處理的任務
        progress = (currentTask / totalTasks) * 100; % 計算當前任務進度百分比

        % 輸出進度資訊
        fprintf('正在處理文件: %s, 獎牌類型: %s, 進度: %.2f%%\n', ...
            cell2mat(uniqueList(fileIdx)), typelist(i), progress);

        medaltype = typelist(i); % 獲取當前的獎牌類型
        
        % 讀取該運動項目的訓練數據（運動員資料）
        fileName = uniqueList(fileIdx);  % 當前文件的名稱
        filePath = fullfile('../dataset/clean/TrainData', cell2mat(fileName));  % 訓練數據的路徑
        summerOly_athletes = readtable(filePath);
        
        % 讀取該運動項目的測試數據
        filePath_test = fullfile('../dataset/clean/TestData', cell2mat(fileName));  % 測試數據的路徑
        summerOly_athletes_test = readtable(filePath_test);  % 讀取測試數據表格
        
        numColumns = size(summerOly_athletes, 2);  % 獲取運動員資料的列數
        newColumnNames = strings(1, numColumns);  % 創建一個字串陣列以存儲新的列名
        for j = 1:numColumns
            newColumnNames(j) = "x" + j;  % 為每列生成新的列名，例如 x1, x2, x3, ...
        end
        summerOly_athletes.Properties.VariableNames = newColumnNames; % 更新訓練數據表格的列名
        summerOly_athletes_test.Properties.VariableNames = newColumnNames; % 更新測試數據表格的列名
        
        %% 篩選獎牌數據
        % 根據獎牌類型篩選訓練數據和測試數據
        filteredTable = summerOly_athletes(strcmp(summerOly_athletes.x1, medaltype), :);  % 篩選出符合條件的訓練數據
        filteredTable_test = summerOly_athletes_test(strcmp(summerOly_athletes_test.x1, medaltype), :);  % 篩選出符合條件的測試數據
        filteredTable_test = sortrows(filteredTable_test, 'x3');  % 根據運動員名稱（是 'x3' 列）排序測試數據
        
        %% 選擇特徵和目標變數
        % 特徵從第5列開始，目標變數是第2列
        X = filteredTable{:, 5:end};  % 訓練數據的特徵（第5列到最後一列）
        Y = filteredTable{:, 2};  % 訓練數據的目標變數（第2列）
        
        %% 數據標準化（Z-score標準化）
        [X_scaled, mu, sigma] = zscore(X);  % 對特徵數據進行 Z-score 標準化
        
        % 標準化測試集特徵
        X_test = filteredTable_test{:, 5:end};  % 測試數據的特徵
        X_test_scaled = (X_test - mu) ./ sigma;  % 使用訓練集的均值和標準差對測試集進行標準化
        
        %% 訓練多個模型並選擇最佳模型
        numModels = 50;  % 訓練 50 個模型
        bestModel = [];  % 初始化最佳模型
        bestRMSE = Inf;  % 初始化最佳 RMSE（均方根誤差），初始設置為無窮大
        
        for train_num = 1:numModels  % 迴圈訓練多個模型
            
            rng(train_num);  % 設置隨機數種子，確保每次訓練結果可重現
            
            %% 劃分訓練集和驗證集
            cv = cvpartition(size(X_scaled, 1), 'HoldOut', 1/3);  % 以 2:1 的比例將數據劃分為訓練集和驗證集
            
            % 劃分訓練集和驗證集
            X_train = X_scaled(training(cv), :);  % 訓練集特徵
            Y_train = Y(training(cv));  % 訓練集目標變數
            X_val = X_scaled(test(cv), :);  % 驗證集特徵
            Y_val = Y(test(cv));  % 驗證集目標變數
            
            % 訓練線性回歸模型
            model = fitlm(X_train, Y_train);  % 使用線性回歸模型進行訓練
            
            % 驗證集預測
            Y_val_pred = predict(model, X_val);  % 在驗證集上進行預測
            Y_val_pred_int = round(Y_val_pred);  % 對預測結果取整數
            
            % 計算驗證集的 RMSE
            mse_val = mean((Y_val - Y_val_pred_int).^2);  % 計算均方誤差
            rmse_val = sqrt(mse_val);  % 計算均方根誤差
            
            % 選擇 RMSE 最小的模型
            if rmse_val < bestRMSE
                bestRMSE = rmse_val;  % 更新最佳 RMSE
                bestModel = model;  % 更新最佳模型
            end
        end
        
        %% 使用最佳模型在測試集上測試
        Y_test_pred = predict(bestModel, X_test_scaled);  % 在測試集上進行預測
        Y_test_pred_int = round(Y_test_pred);  % 取整數
        
        % 計算預測結果的信賴區間
        [Y_test_pred, Y_test_pred_ci] = predict(bestModel, X_test_scaled, 'Prediction', 'curve', 'Alpha', alpha);  % 計算 95% 信賴區間
        Y_test_pred_lower = Y_test_pred_ci(:, 1);  % 信賴區間下限
        Y_test_pred_upper = Y_test_pred_ci(:, 2);  % 信賴區間上限
        
        % 生成列名：運動名稱 + 獎牌類型
        columnName = [char(fileName), '_', char(medaltype)];
        
        lowerColumnName = [columnName, '_Lower'];  % 信賴區間下限列名
        upperColumnName = [columnName, '_Upper'];  % 信賴區間上限列名
        
        resultsTable.(columnName) = Y_test_pred_int;  % 保存預測結果到結果表格
        
        % 保存信賴區間到信賴區間表格
        confidenceIntervals.(lowerColumnName) = Y_test_pred_lower;
        confidenceIntervals.(upperColumnName) = Y_test_pred_upper;
        
        % 獲取線性回歸模型的權重（係數）
        coefficients = model.Coefficients.Estimate;  % 獲取模型係數
        
        % 保存權重到權重表格
        weightsTable.(columnName) = coefficients;  % 將係數保存到表格
    end
end

%% 結果表格處理
% 添加 'Country' 列，將測試數據中的運動員名稱作為國家信息
resultsTable.Country = filteredTable_test.x3;

% 獲取所有列名，排除 'Country' 列
columnsToProcess = setdiff(resultsTable.Properties.VariableNames, 'Country');

% 遍歷所有需要處理的列
for i = 1:length(columnsToProcess)
    columnName = columnsToProcess{i};  % 當前列的列名
    
    % 獲取當前列的數據
    columnData = resultsTable.(columnName);
    
    % 將 NaN 值替換為 0
    columnData(isnan(columnData)) = 0;
    
    % 將負數值替換為 0
    columnData(columnData < 0) = 0;
    
    % 將處理後的數據寫回到結果表格
    resultsTable.(columnName) = columnData;
end

% 計算金牌數量（金牌數據位於第1、5、8...列）
goldColumns = 1:4:188;  % 生成金牌數據的列索引（每4列的第1列）
resultsTable.Gold_Medal_Count = sum(resultsTable{:, goldColumns}, 2);  % 計算每一行（即每個國家）的金牌數量

% 計算銀牌數量（銀牌數據位於第2、6、9...列）
silverColumns = 2:4:188;  % 生成銀牌數據的列索引（每4列的第2列）
resultsTable.Silver_Medal_Count = sum(resultsTable{:, silverColumns}, 2);  % 計算每一行（即每個國家）的銀牌數量

% 計算銅牌數量（銅牌數據位於第3、7、10...列）
bronzeColumns = 3:4:188;  % 生成銅牌數據的列索引（每4列的第3列）
resultsTable.Bronze_Medal_Count = sum(resultsTable{:, bronzeColumns}, 2);  % 計算每一行（即每個國家）的銅牌數量

% 計算預測的總獎牌數量（預測數據位於第4、8、11...列）
predictedColumns = 4:4:188;  % 生成預測數據的列索引（每4列的第4列）
resultsTable.Predicted_Total_Medals = sum(resultsTable{:, predictedColumns}, 2);  % 計算每一行（即每個國家）的總預測獎牌數量

% 計算加和後的總獎牌數量（由金、銀、銅牌數量相加得來）
resultsTable.Summed_Total_Medals = resultsTable.Gold_Medal_Count + ...
    resultsTable.Silver_Medal_Count + ...
    resultsTable.Bronze_Medal_Count;

% 保存結果到 CSV 文件
outputFile = 'step2_output_多元線性回歸預測結果.csv';
writetable(resultsTable, outputFile);
fprintf('所有結果已保存到文件: %s\n', outputFile);

%% 信賴區間處理
% 添加 'Country' 列，將測試數據中的運動員名稱（是 'x3' 列）作為國家信息
confidenceIntervals.Country = filteredTable_test.x3;

% 獲取所有列名，排除 'Country' 列
columnsToProcess = setdiff(confidenceIntervals.Properties.VariableNames, 'Country');

% 遍歷所有需要處理的列
for i = 1:length(columnsToProcess)
    columnName = columnsToProcess{i};  % 當前列的列名
    
    % 獲取當前列的數據
    columnData = confidenceIntervals.(columnName);
    
    % 將 NaN 值替換為 0
    columnData(isnan(columnData)) = 0;
    
    % 將負數值替換為 0
    % columnData(columnData < 0) = 0;
    
    % 將處理後的數據寫回到信賴區間表格
    confidenceIntervals.(columnName) = columnData;
end

% 計算金牌信賴區間的下限和上限
goldColumns = 1:8:376;  % 生成金牌列的索引（每8列的第1列是金牌的下限）
confidenceIntervals.Gold_Medal_Count_Lower = sum(confidenceIntervals{:, goldColumns}, 2);  % 計算金牌的信賴區間下限

goldColumns = 2:8:376;  % 生成金牌列的索引（每8列的第2列是金牌的上限）
confidenceIntervals.Gold_Medal_Count_Upper = sum(confidenceIntervals{:, goldColumns}, 2);  % 計算金牌的信賴區間上限

% 計算銀牌信賴區間的下限和上限
silverColumns = 3:8:376;  % 生成銀牌列的索引（每8列的第3列是銀牌的下限）
confidenceIntervals.Silver_Medal_Count_Lower = sum(confidenceIntervals{:, silverColumns}, 2);  % 計算銀牌的信賴區間下限

silverColumns = 4:8:376;  % 生成銀牌列的索引（每8列的第4列是銀牌的上限）
confidenceIntervals.Silver_Medal_Count_Upper = sum(confidenceIntervals{:, silverColumns}, 2);  % 計算銀牌的信賴區間上限

% 計算銅牌信賴區間的下限和上限
bronzeColumns = 5:8:376;  % 生成銅牌列的索引（每8列的第5列是銅牌的下限）
confidenceIntervals.Bronze_Medal_Count_Lower = sum(confidenceIntervals{:, bronzeColumns}, 2);  % 計算銅牌的信賴區間下限

bronzeColumns = 6:8:376;  % 生成銅牌列的索引（每8列的第6列是銅牌的上限）
confidenceIntervals.Bronze_Medal_Count_Upper = sum(confidenceIntervals{:, bronzeColumns}, 2);  % 計算銅牌的信賴區間上限

% 計算預測總獎牌信賴區間的下限和上限
predictedColumns = 7:8:376;  % 生成預測列的索引（每8列的第7列是預測總獎牌的下限）
confidenceIntervals.Predicted_Total_Medals_Lower = sum(confidenceIntervals{:, predictedColumns}, 2);  % 計算總獎牌的信賴區間下限

predictedColumns = 8:8:376;  % 生成預測列的索引（每8列的第8列是預測總獎牌的上限）
confidenceIntervals.Predicted_Total_Medals_Upper = sum(confidenceIntervals{:, predictedColumns}, 2);  % 計算總獎牌的信賴區間上限

% 計算加和後的總獎牌信賴區間（下限與上限）
confidenceIntervals.Summed_Total_Medals_Lower = confidenceIntervals.Gold_Medal_Count_Lower + ...
    confidenceIntervals.Silver_Medal_Count_Lower + ...
    confidenceIntervals.Bronze_Medal_Count_Lower;  % 計算總獎牌信賴區間下限

confidenceIntervals.Summed_Total_Medals_Upper = confidenceIntervals.Gold_Medal_Count_Upper + ...
    confidenceIntervals.Silver_Medal_Count_Upper + ...
    confidenceIntervals.Bronze_Medal_Count_Upper;  % 計算總獎牌信賴區間上限

% 保存信賴區間結果到 CSV 文件
outputFile = 'step2_output_多元線性回歸預測信賴區間.csv';
writetable(confidenceIntervals, outputFile);
fprintf('所有結果已保存到文件: %s\n', outputFile);

%% 保存權重係數結果
outputFile = 'step2_output_多元線性回歸預測權重係數.csv';
writetable(weightsTable, outputFile);
fprintf('所有結果已保存到文件: %s\n', outputFile);

