clear;
clc;
warning('off');
rng(123);

%% 讀取需要預測數據
fileName = 'step1_output_PredictionData.csv'; % csv 檔路徑
data = readtable(fileName); % 讀取整個表格

% 提取第一列資料
firstColumn = data{:, 1};

% 去除重複值
uniqueList = unique(firstColumn); % 去除重複值並排序
numFiles = length(uniqueList);

% 獎牌類別
typelist = ["Gold", "Silver", "Bronze", "All"];

resultsTable = table();

weightTable = table();

% 初始化信賴區間表
confidenceIntervals = table();

% 設置信賴水準
alpha = 0.05; % 95% 信賴區間

totalTasks = numFiles * length(typelist);

% 當前任務計數器
currentTask = 0;

%% 迴圈處理每個檔
for fileIdx = 1:numFiles

    for i = 1:length(typelist)

        rng(123);

        currentTask = currentTask + 1; % 更新任務計數器
        progress = (currentTask / totalTasks) * 100; % 計算進度百分比

        % 列印進度資訊
        fprintf('正在處理文件: %s, 獎牌類型: %s, 進度: %.2f%%\n', ...
                cell2mat(uniqueList(fileIdx)), typelist(i), progress);
        
        medaltype = typelist(i); % 獲取當前元素

        fileName = uniqueList(fileIdx);
        filePath = fullfile('../dataset/clean/TrainData', cell2mat(fileName));
        summerOly_athletes = readtable(filePath);
        
        filePath_test = fullfile('../dataset/clean/TestData', cell2mat(fileName));
        summerOly_athletes_test = readtable(filePath_test);
    
        numColumns = size(summerOly_athletes, 2);
        newColumnNames = strings(1, numColumns); % 創建一個字串陣列
        for j = 1:numColumns
            newColumnNames(j) = "x" + j; % 生成 x1, x2, x3, ...
        end
        summerOly_athletes.Properties.VariableNames = newColumnNames; % 獲取列名
        summerOly_athletes_test.Properties.VariableNames = newColumnNames; % 獲取列名
        
        %% 篩選獎牌數據
        filteredTable = summerOly_athletes(strcmp(summerOly_athletes.x1, medaltype), :);
        filteredTable_test = summerOly_athletes_test(strcmp(summerOly_athletes_test.x1, medaltype), :);
        filteredTable_test = sortrows(filteredTable_test, 'x3');
        
        %% 選擇特徵和目標變數
        X = filteredTable{:, 5:end}; % 特徵從第5列開始
        Y = filteredTable{:, 2};
        
        %% 數據標準化（Z-score標準化）
        [X_scaled, mu, sigma] = zscore(X);
        
        X_test = filteredTable_test{:, 5:end};
        X_test_scaled = (X_test - mu) ./ sigma;

        %% LASSO
        [B, FitInfo] = lasso(X_scaled, Y, 'CV', 10);

        % 選擇最優的 Lambda 值
        Lambda1SE = FitInfo.Lambda1SE; % 最小均方誤差加一個標準差的 Lambda 值
        LambdaMinMSE = FitInfo.LambdaMinMSE; % 最小均方誤差的 Lambda 值

        % 使用最優的 Lambda 值進行預測
        Y_test_pred = X_test_scaled * B(:, FitInfo.Index1SE); % 使用 Lambda1SE 進行預測
        Y_test_pred_int = round(Y_test_pred);

        % 生成列名：運動名稱 + 獎牌類型
        columnName = [char(fileName), '_', char(medaltype)];

        resultsTable.(columnName) = Y_test_pred_int;

        if strcmp(medaltype, 'All')
            weightTable.(columnName) = B(:, 51);
        end

    end
end

%%
resultsTable.Country = filteredTable_test.x3;

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

% 計算金牌數量（第 1、5、8... 列的和）
goldColumns = 1:4:188;  % 生成金牌列的索引
resultsTable.Gold_Medal_Count = sum(resultsTable{:, goldColumns}, 2);

% 計算銀牌數量（第 2、6、9... 列的和）
silverColumns = 2:4:188;  % 生成銀牌列的索引
resultsTable.Silver_Medal_Count = sum(resultsTable{:, silverColumns}, 2);

% 計算銅牌數量（第 3、7、10... 列的和）
bronzeColumns = 3:4:188;  % 生成銅牌列的索引
resultsTable.Bronze_Medal_Count = sum(resultsTable{:, bronzeColumns}, 2);

% 計算預測的總獎牌數量（第 4、8、11... 列的和）
predictedColumns = 4:4:188;  % 生成預測列的索引
resultsTable.Predicted_Total_Medals = sum(resultsTable{:, predictedColumns}, 2);

% 計算加和求得的總獎牌數量
resultsTable.Summed_Total_Medals = resultsTable.Gold_Medal_Count + ...
                                   resultsTable.Silver_Medal_Count + ...
                                   resultsTable.Bronze_Medal_Count;

% 保存結果到 csv 檔
outputFile = 'step3_output_LASSO回歸預測結果.csv';
writetable(resultsTable, outputFile);
fprintf('所有結果已保存到檔: %s\n', outputFile);

%%
outputFile = 'step3_output_LASSO回歸權重資訊.csv';
writetable(weightTable, outputFile);
fprintf('所有結果已保存到文件: %s\n', outputFile);

