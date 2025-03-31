clear; % 清除所有變數
clc; % 清除命令視窗
warning('off'); % 關閉警告
rng(123); % 設定隨機種子，以確保結果可重現

%% 讀取需要預測數據
fileName = 'step1_output_PredictionData.csv'; % 設定 CSV 檔案名稱
data = readtable(fileName); % 讀取 CSV 檔案內容，轉換為表格格式

% 提取第一列資料（運動項目名稱）
firstColumn = data{:, 1};

% 去除重複值，獲取所有唯一的運動項目
uniqueList = unique(firstColumn); 
numFiles = length(uniqueList); % 計算不同運動項目的數量

% 定義獎牌類別
typelist = ["Gold", "Silver", "Bronze", "All"];

% 初始化存儲結果的表格
resultsTable = table(); 
confidenceIntervals = table(); % 存儲信賴區間
weightsTable = table(); % 存儲模型的權重參數

% 設定信賴區間的信心水準
alpha = 0.05; % 95% 信賴區間

% 計算總任務數量（運動項目數 * 獎牌類型數）
totalTasks = numFiles * length(typelist);
currentTask = 0; % 初始化當前任務計數器

%% 迴圈處理每個檔
for fileIdx = 1:numFiles
    for i = 1:length(typelist)
        rng(123); % 重設隨機種子，確保可重現性

        currentTask = currentTask + 1; % 更新當前處理的任務計數
        progress = (currentTask / totalTasks) * 100; % 計算進度百分比

        % 顯示處理進度
        fprintf('正在處理文件: %s, 獎牌類型: %s, 進度: %.2f%%\n', ...
                cell2mat(uniqueList(fileIdx)), typelist(i), progress);
        
        % 獲取當前運動項目和獎牌類型
        medaltype = typelist(i);
        fileName = uniqueList(fileIdx);

        % 讀取該運動項目的訓練數據與測試數據
        filePath = fullfile('../dataset/clean/TrainData', cell2mat(fileName));
        summerOly_athletes = readtable(filePath);
        
        filePath_test = fullfile('../dataset/clean/TestData', cell2mat(fileName));
        summerOly_athletes_test = readtable(filePath_test);
    
        % 統一欄位名稱，將所有列命名為 x1, x2, x3, ...
        numColumns = size(summerOly_athletes, 2);
        newColumnNames = strings(1, numColumns); % 建立字串陣列
        for j = 1:numColumns
            newColumnNames(j) = "x" + j;
        end
        summerOly_athletes.Properties.VariableNames = newColumnNames;
        summerOly_athletes_test.Properties.VariableNames = newColumnNames;
        
        %% 篩選獎牌數據
        filteredTable = summerOly_athletes(strcmp(summerOly_athletes.x1, medaltype), :); % 根據獎牌類型篩選數據
        filteredTable_test = summerOly_athletes_test(strcmp(summerOly_athletes_test.x1, medaltype), :);
        filteredTable_test = sortrows(filteredTable_test, 'x3'); % 根據第3列（例如國家名稱或其他）排序
        
        %% 選擇特徵和目標變數
        X = filteredTable{:, 5:end}; % 特徵從第5列開始
        Y = filteredTable{:, 2}; % 目標變數（可能是某種指標，如獲得的金牌數量）

        %% 數據標準化（Z-score標準化）
        [X_scaled, mu, sigma] = zscore(X); % 標準化數據

        X_test = filteredTable_test{:, 5:end};
        X_test_scaled = (X_test - mu) ./ sigma; % 測試數據標準化
        
        %% 訓練多個隨機森林模型並選擇最佳模型
        % 初始化變數
        % 訓練多個模型並選擇最佳模型
        numModels = 10; % 訓練模型的次數
        bestModel = [];
        bestRMSE = Inf;
        
        for train_num = 1:numModels
            rng(train_num);

            % 分割訓練集和驗證集 (2:1)
            cv = cvpartition(size(X_scaled, 1), 'HoldOut', 1/2);
            X_train = X_scaled(training(cv), :);
            Y_train = Y(training(cv));
            X_val = X_scaled(test(cv), :);
            Y_val = Y(test(cv));

            % 訓練隨機森林模型
            numTrees = 100;
            model = TreeBagger(numTrees, X_train, Y_train, 'Method', 'regression');

            % 在驗證集上評估
            Y_val_pred = predict(model, X_val);
            Y_val_pred_int = round(Y_val_pred);
            mse_val = mean((Y_val - Y_val_pred_int).^2);
            rmse_val = sqrt(mse_val);

            % 選擇 RMSE 最小的模型
            if rmse_val < bestRMSE
                bestRMSE = rmse_val;
                bestModel = model;
            end
        end
        
        disp(['最佳模型的驗證集 RMSE: ', num2str(bestRMSE)]);
        
        % 使用最佳模型在測試集上測試
        Y_test_pred = predict(bestModel, X_test_scaled);
        Y_test_pred_int = round(Y_test_pred); % 將預測結果四捨五入為整數
        
        % 生成列名：運動名稱 + 獎牌類型
        columnName = [char(fileName), '_', char(medaltype)];
        
        lowerColumnName = [columnName, '_Lower'];
        upperColumnName = [columnName, '_Upper'];
        
        resultsTable.(columnName) = Y_test_pred_int; % 保存預測結果

    end
end

%% 處理結果數據
resultsTable.Country = filteredTable_test.x3; % 追加國家名稱列

% 獲取所有列名（除了 Country 列）
columnsToProcess = setdiff(resultsTable.Properties.VariableNames, 'Country');

% 遍歷所有需要處理的列
for i = 1:length(columnsToProcess)
    columnName = columnsToProcess{i};

    % 獲取當前列資料
    columnData = resultsTable.(columnName);

    % 將 NaN 替換為 0
    columnData(isnan(columnData)) = 0;

    % 將負數替換為 0
    columnData(columnData < 0) = 0;

    % 將處理後的資料寫回 resultsTable
    resultsTable.(columnName) = columnData;
end

% 計算金牌數量（金牌數據位於第1、5、8...列）的和
goldColumns = 1:4:numFiles*4;  % 生成金牌列的索引
resultsTable.Gold_Medal_Count = sum(resultsTable{:, goldColumns}, 2);

% 計算銀牌數量（銀牌數據位於第2、6、9...列）的和
silverColumns = 2:4:numFiles*4;  % 生成銀牌列的索引
resultsTable.Silver_Medal_Count = sum(resultsTable{:, silverColumns}, 2);

% 計算銅牌數量（銅牌數據位於第3、7、10...列）的和
bronzeColumns = 3:4:numFiles*4;  % 生成銅牌列的索引
resultsTable.Bronze_Medal_Count = sum(resultsTable{:, bronzeColumns}, 2);

% 計算預測的總獎牌數量（預測數據位於第4、8、11...列）的和
predictedColumns = 4:4:numFiles*4;  % 生成預測列的索引
resultsTable.Predicted_Total_Medals = sum(resultsTable{:, predictedColumns}, 2);

% 計算加和求得的總獎牌數量
resultsTable.Summed_Total_Medals = resultsTable.Gold_Medal_Count + ...
                                   resultsTable.Silver_Medal_Count + ...
                                   resultsTable.Bronze_Medal_Count;

% 保存結果到 csv 檔
outputFile = 'step2_output_隨機森林預測結果.csv';
writetable(resultsTable, outputFile);
fprintf('所有結果已保存到文件: %s\n', outputFile);
