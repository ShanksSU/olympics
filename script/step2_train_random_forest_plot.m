clear; % 清除所有變數
clc; % 清除命令視窗
warning('off'); % 關閉警告
rng(123); % 設定隨機種子，以確保結果可重現

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

% 初始化信賴區間表
confidenceIntervals = table();

% 初始化權重參數表
weightsTable = table();

% 設置信賴水準
alpha = 0.05; % 95% 信賴區間

totalTasks = numFiles * length(typelist);

% 當前任務計數器
currentTask = 0;

% numFiles = 1;

%% 迴圈處理每個檔
for fileIdx = 1:numFiles

    for i = 1:length(typelist)

        rng(123);

        currentTask = currentTask + 1; % 更新任務計數器
        progress = (currentTask / totalTasks) * 100; % 計算進度百分比

        % 輸出進度資訊
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

        % 獲取運動類別和獎牌類別
        sportName = cell2mat(fileName); % 運動類別
        medalType = char(medaltype);    % 獎牌類別
        
        %% 篩選獎牌數據
        filteredTable = summerOly_athletes(strcmp(summerOly_athletes.x1, medaltype), :);
        filteredTable_test = summerOly_athletes_test(strcmp(summerOly_athletes_test.x1, medaltype), :);
        filteredTable_test = sortrows(filteredTable_test, 'x3');
        
        % 創建保存圖片的資料夾
        outputFolder = '隨機森林結果圖';
        if ~exist(outputFolder, 'dir')
            mkdir(outputFolder);
        end
        
        %% 選擇特徵和目標變數
        X = filteredTable{:, 5:end}; % 特徵從第5列開始
        Y = filteredTable{:, 2};
        
        %% 數據標準化（Z-score標準化）
        [X_scaled, mu, sigma] = zscore(X);
        
        X_test = filteredTable_test{:, 5:end};
        X_test_scaled = (X_test - mu) ./ sigma;
        
        %% 訓練多個隨機森林模型並選擇最佳模型
        numModels = 10;
        bestModel = [];
        bestRMSE = Inf;
        
        for train_num = 1:numModels
            rng(train_num); % 設置隨機種子以確保可重複性
        
            % 劃分訓練集和驗證集
            cv = cvpartition(size(X_scaled, 1), 'HoldOut', 1/3); % 2:1 的比例
        
            % 劃分訓練集和驗證集
            X_train = X_scaled(training(cv), :);
            Y_train = Y(training(cv));
            X_val = X_scaled(test(cv), :);
            Y_val = Y(test(cv));
        
            % 訓練隨機森林模型
            numTrees = 100; % 設置樹的數量
            model = TreeBagger(numTrees, X_train, Y_train, 'Method', 'regression', ...
                               'OOBPredictorImportance', 'on'); % 啟用特徵重要性計算
        
            % 驗證集預測
            Y_val_pred = predict(model, X_val);
            Y_val_pred_int = round(Y_val_pred);
        
            % 計算驗證集的 RMSE
            mse_val = mean((Y_val - Y_val_pred_int).^2);
            rmse_val = sqrt(mse_val);
        
            % 選擇最佳模型
            if rmse_val < bestRMSE
                bestRMSE = rmse_val;
                bestModel = model;
            end
        end
        
        disp(['最佳模型的驗證集 RMSE: ', num2str(bestRMSE)]);
        
        %% 使用最佳模型進行訓練集和驗證集的預測
        % 訓練集預測
        Y_train_pred = predict(bestModel, X_train);
        Y_train_pred_int = round(Y_train_pred);
        
        % 驗證集預測
        Y_val_pred = predict(bestModel, X_val);
        Y_val_pred_int = round(Y_val_pred);
        
        %% 繪製並保存圖片
        % 1. 訓練集真實值-預測值分佈圖
        figure('Visible', 'off'); % 設置圖形視窗不可見
        scatter(Y_train, Y_train_pred_int);
        hold on;
        plot([min(Y_train), max(Y_train)], [min(Y_train), max(Y_train)], 'r--', 'LineWidth', 2);
        xlabel('真實值');
        ylabel('預測值');
        title(sprintf('訓練集真實值-預測值分佈 (%s - %s)', sportName, medalType));
        saveas(gcf, fullfile(outputFolder, sprintf('訓練集真實值-預測值分佈_%s_%s.png', sportName, medalType)));
        
        % 2. 驗證集真實值-預測值分佈圖
        figure('Visible', 'off'); % 設置圖形視窗不可見
        scatter(Y_val, Y_val_pred_int);
        hold on;
        plot([min(Y_val), max(Y_val)], [min(Y_val), max(Y_val)], 'r--', 'LineWidth', 2);
        xlabel('真實值');
        ylabel('預測值');
        title(sprintf('驗證集真實值-預測值分佈 (%s - %s)', sportName, medalType));
        saveas(gcf, fullfile(outputFolder, sprintf('驗證集真實值-預測值分佈_%s_%s.png', sportName, medalType)));
        
        % 3. 訓練集殘差分佈圖
        figure('Visible', 'off'); % 設置圖形視窗不可見
        residuals_train = Y_train - Y_train_pred_int;
        histogram(residuals_train, 20);
        xlabel('殘差');
        ylabel('頻率');
        title(sprintf('訓練集殘差分佈 (%s - %s)', sportName, medalType));
        saveas(gcf, fullfile(outputFolder, sprintf('訓練集殘差分佈_%s_%s.png', sportName, medalType)));
        
        % 4. 驗證集殘差分佈圖
        figure('Visible', 'off'); % 設置圖形視窗不可見
        residuals_val = Y_val - Y_val_pred_int;
        histogram(residuals_val, 20);
        xlabel('殘差');
        ylabel('頻率');
        title(sprintf('驗證集殘差分佈 (%s - %s)', sportName, medalType));
        saveas(gcf, fullfile(outputFolder, sprintf('驗證集殘差分佈_%s_%s.png', sportName, medalType)));
        
        % 5. 特徵重要性圖
        figure('Visible', 'off'); % 設置圖形視窗不可見
        importance = bestModel.OOBPermutedVarDeltaError; % 獲取特徵重要性
        bar(importance);
        xlabel('特徵索引');
        ylabel('特徵重要性');
        title(sprintf('特徵重要性 (%s - %s)', sportName, medalType));
        saveas(gcf, fullfile(outputFolder, sprintf('特徵重要性_%s_%s.png', sportName, medalType)));
        
        %% 使用最佳模型在測試集上測試
        Y_test_pred = predict(bestModel, X_test_scaled);
        Y_test_pred_int = round(Y_test_pred);
        
        % 生成列名：運動名稱 + 獎牌類型
        columnName = [char(fileName), '_', char(medaltype)];
        
        lowerColumnName = [columnName, '_Lower'];
        upperColumnName = [columnName, '_Upper'];
        
        resultsTable.(columnName) = Y_test_pred_int;

    end
end
