clear;
clc;
warning('off');
rng(123);

%% 讀取需要預測的大類
fileName = 'step1_output_PredictionData.csv'; % csv 檔路徑
data = readtable(fileName); % 讀取整個表格

% 提取第一列資料
firstColumn = data{:, 1};

% 去除重複值
uniqueList = unique(firstColumn); % 去除重複值並排序
numFiles = length(uniqueList);

% 獎牌類別
typelist = "All";

resultsTable = table();

% 初始化信賴區間表
probTable = table();

% 設置信賴水準
alpha = 0.05; % 95% 信賴區間

totalTasks = numFiles * length(typelist);

% 當前任務計數器
currentTask = 0;

weightTable = readtable('step3_output_LASSO回歸權重資訊.csv');

%% 迴圈處理每個檔
for fileIdx = 1:numFiles

    rng(123);

    currentTask = currentTask + 1; % 更新任務計數器
    progress = (currentTask / totalTasks) * 100; % 計算進度百分比

    % 列印進度資訊
    fprintf('正在處理文件: %s, 獎牌類型: %s, 進度: %.2f%%\n', ...
            cell2mat(uniqueList(fileIdx)), 'All', progress);
    
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
    filteredTable = summerOly_athletes(strcmp(summerOly_athletes.x1, typelist), :);
    filteredTable_test = summerOly_athletes_test(strcmp(summerOly_athletes_test.x1, typelist), :);
    filteredTable_test = sortrows(filteredTable_test, 'x3'); 

    %% 選擇特徵和目標變數
    X = filteredTable{:, 5:end}; % 特徵從第5列開始
    Y = filteredTable{:, 2};
    Y(Y ~= 0) = 1;

    weightvalues = weightTable{:, fileIdx};

    % 找到需要刪除的列索引
    columnsToDelete = find(weightvalues == 0); % 假設值為 1 的列需要被刪除
    
    % 刪除 X 中對應的列
    X(:, columnsToDelete) = [];

    %% 數據標準化（Z-score標準化）
    [X_scaled, mu, sigma] = zscore(X);

    X_test = filteredTable_test{:, 5:end};

    % 刪除 X_test 中對應的列
    X_test(:, columnsToDelete) = [];

    X_test_scaled = (X_test - mu) ./ sigma;

    %% 劃分訓練集和驗證集
    cv = cvpartition(size(X_scaled, 1), 'HoldOut', 1/3); % 2:1 的比例

    %% 訓練多個模型並選擇最佳模型
    numModels = 50;
    bestModel = [];
    bestAccuracy = -Inf; % 初始化為負無窮，因為我們希望最大化準確率

    for train_num = 1:numModels
        rng(train_num);

        % 劃分訓練集和驗證集
        X_train = X_scaled(training(cv), :);
        Y_train = Y(training(cv));
        X_val = X_scaled(test(cv), :);
        Y_val = Y(test(cv));

        % 訓練模型
        model = fitglm(X_train, Y_train, 'Distribution', 'binomial');

        % 驗證集預測
        Y_val_pred_prob = predict(model, X_val); % 預測概率
        Y_val_pred = double(Y_val_pred_prob > 0.5); % 將概率轉換為二分類標籤 (0 或 1)

        % 計算驗證集的準確率
        accuracy_val = sum(Y_val == Y_val_pred) / length(Y_val);
    
        % 選擇最佳模型
        if accuracy_val > bestAccuracy
            bestAccuracy = accuracy_val;
            bestModel = model;
        end
    end

    disp(['最佳模型的驗證集準確率: ', num2str(bestAccuracy * 100), '%']);

    %% 使用最佳模型在測試集上測試
    Y_test_pred_prob = predict(bestModel, X_test_scaled); % 預測概率
    Y_test_pred = double(Y_test_pred_prob > 0.25); % 將概率轉換為二分類標籤 (0 或 1)

    % 生成列名：運動名稱 + 獎牌類型
    columnName = [char(fileName), '_All'];

    resultsTable.(columnName) = Y_test_pred;

    % 將預測概率保存到表中
    probTable.(columnName) = Y_test_pred_prob;

end

%%
resultsTable.Country = filteredTable_test.x3;

colNames = resultsTable.Properties.VariableNames;
numericCols = colNames(1:end-1);
resultsTable.sum = sum(table2array(resultsTable(:, numericCols)), 2);

probTable.Country = filteredTable_test.x3;

%%
non_conutries = readtable('../dataset/clean/No_Medals/No_Medals_2016_2024.xlsx');

% 讀取 non_countries 表中的國家名稱
non_countries_list = non_conutries.MissingCountries;

% 篩選 resultsTable 中的行，其中 Country 列的值在 non_countries_list 中
filtered_resultsTable = resultsTable(ismember(resultsTable.Country, non_countries_list), :);

filtered_probTable = probTable(ismember(probTable.Country, non_countries_list), :);

%% 保存結果到 csv 檔
outputFile = 'step4_output_Logistic_regression預測結果.csv';
writetable(resultsTable, outputFile);
fprintf('所有結果已保存到文件: %s\n', outputFile);

outputFile = 'step4_output_Logistic_regression預測概率.csv';
writetable(probTable, outputFile);
fprintf('所有結果已保存到文件: %s\n', outputFile);

outputFile = 'step4_output_Logistic_regression預測結果_篩選後.csv';
writetable(filtered_resultsTable, outputFile);
fprintf('所有結果已保存到文件: %s\n', outputFile);

outputFile = 'step4_output_Logistic_regression預測概率_篩選後.csv';
writetable(filtered_probTable, outputFile);
fprintf('所有結果已保存到文件: %s\n', outputFile);

