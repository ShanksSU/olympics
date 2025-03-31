clear;  % 清除工作空間中的所有變量
clc;    % 清除命令窗口中的內容
warning('off');  % 關閉所有警告顯示
rng(123);  % 設定隨機數種子為123，保證實驗的可重複性

%% 讀取需要預測數據
fileName = 'step1_output_PredictionData.csv';  % 定義要讀取的CSV文件路徑
data = readtable(fileName);  % 使用 readtable 函數讀取CSV文件，並將其存儲為表格形式

% 提取第一列資料
firstColumn = data{:, 1};  % 提取表格中的第一列數據並存儲到 firstColumn 變量中

% 去除重複值
uniqueList = unique(firstColumn);  % 使用 unique 函數去除 firstColumn 中的重複值，並返回唯一的值列表
numFiles = length(uniqueList);  % 計算 uniqueList 中元素的數量，即不同檔案的數量

% 獎牌類別
typelist = ["Gold", "Silver", "Bronze", "All"];  % 定義一個包含獎牌類型的數組：金牌、銀牌、銅牌、全部

resultsTable = table();  % 初始化一個空的表格，用於存儲最終的預測結果

% 初始化信賴區間表
confidenceIntervals = table();  % 初始化一個空的表格，用於存儲預測的信賴區間

% 設置信賴水準
alpha = 0.05;  % 設定信賴水準為 0.05，對應 95% 的信賴區間

totalTasks = 1 * 1;  % 設定總任務數量，這裡設為 1，假設有 1 個檔案需要處理，這可能是需要根據情況修改

% 當前任務計數器
currentTask = 0;  % 初始化當前任務計數器，該計數器用於跟踪進度

%% 迴圈處理每個檔
for fileIdx = 5:5  % 迴圈處理第5個檔案（您可以根據需要修改索引範圍）
    
    for i = 1:1  % 迴圈處理第一個獎牌類型（可以擴展為多個獎牌類型）
        
        rng(123);  % 設置隨機數種子，以保證每次運行結果一致
        
        currentTask = currentTask + 1; % 更新任務計數器
        progress = (currentTask / totalTasks) * 100; % 計算進度百分比
        
        % 輸出進度資訊
        fprintf('正在處理文件: %s, 獎牌類型: %s, 進度: %.2f%%\n', ...
            cell2mat(uniqueList(fileIdx)), typelist(i), progress);
        
        medaltype = typelist(i); % 獲取當前處理的獎牌類型
        
        % 訓練數據和測試數據的路徑
        fileName = uniqueList(fileIdx);
        filePath = fullfile('../dataset/clean/TrainData', cell2mat(fileName)); % 訓練數據檔案路徑
        summerOly_athletes = readtable(filePath); % 讀取訓練數據
        
        filePath_test = fullfile('../dataset/clean/TestData', cell2mat(fileName)); % 測試數據檔案路徑
        summerOly_athletes_test = readtable(filePath_test); % 讀取測試數據
        
        % 重新命名列以方便操作
        numColumns = size(summerOly_athletes, 2);
        newColumnNames = strings(1, numColumns); % 創建一個字串陣列
        for j = 1:numColumns
            newColumnNames(j) = "x" + j; % 生成 x1, x2, x3, ...
        end
        summerOly_athletes.Properties.VariableNames = newColumnNames; % 獲取列名
        summerOly_athletes_test.Properties.VariableNames = newColumnNames; % 獲取測試數據的列名
        
        %% 篩選獎牌數據
        % 根據獎牌類型過濾訓練數據和測試數據
        filteredTable = summerOly_athletes(strcmp(summerOly_athletes.x1, medaltype), :); % 選擇特定獎牌類型
        filteredTable_test = summerOly_athletes_test(strcmp(summerOly_athletes_test.x1, medaltype), :); % 測試數據過濾
        filteredTable_test = sortrows(filteredTable_test, 'x3'); % 排序測試數據
        
        %% 選擇特徵和目標變數
        X = filteredTable{:, 5:end}; % 特徵從第5列開始
        Y = filteredTable{:, 2}; % 目標變數為第二列（獎牌情況）
        
        %% 數據標準化（Z-score標準化）
        [X_scaled, mu, sigma] = zscore(X); % 計算標準化後的特徵
        
        % 對測試數據進行標準化
        X_test = filteredTable_test{:, 5:end};
        X_test_scaled = (X_test - mu) ./ sigma; % 測試數據標準化
        
        % 繪製原始特徵與標準化特徵的對比圖
        figure;
        subplot(1,2,1);
        boxplot(X);  % 顯示原始特徵的箱型圖
        title('原始特徵分佈');
        xlabel('特徵');
        ylabel('數值');
        subplot(1,2,2);
        boxplot(X_scaled);  % 顯示標準化後的箱型圖
        title('標準化後特徵分佈');
        xlabel('特徵');
        ylabel('數值');
        saveas(gcf, 'step2_特徵分佈.png');  % 保存圖形
        
        % 記錄每輪RMSE的變化
        numModels = 50;  % 訓練50個模型
        train_rmse = zeros(numModels, 1);  % 用於存儲訓練集RMSE的數組
        val_rmse = zeros(numModels, 1);    % 用於存儲驗證集RMSE的數組
        
        %% 訓練多個模型並選擇最佳模型
        bestModel = [];  % 初始化最佳模型
        bestRMSE = Inf;  % 設定一個初始的無窮大RMSE
        
        for train_num = 1:numModels
            rng(train_num);  % 設置每次訓練的隨機數種子
            
            %% 劃分訓練集和驗證集
            cv = cvpartition(size(X_scaled, 1), 'HoldOut', 1/3); % 2:1 的比例
            X_train = X_scaled(training(cv), :);  % 訓練集特徵
            Y_train = Y(training(cv));            % 訓練集目標變數
            X_val = X_scaled(test(cv), :);        % 驗證集特徵
            Y_val = Y(test(cv));                  % 驗證集目標變數
            
            % 訓練模型
            model = fitlm(X_train, Y_train);  % 使用線性回歸模型進行訓練
            
            % 驗證集預測
            Y_val_pred = predict(model, X_val);  % 預測驗證集
            Y_val_pred_int = round(Y_val_pred);  % 四捨五入獲得整數預測結果
            
            % 計算驗證集的 RMSE
            mse_val = mean((Y_val - Y_val_pred_int).^2);
            rmse_val = sqrt(mse_val);
            
            % 記錄訓練和驗證集的RMSE
            train_rmse(train_num) = sqrt(mean((Y_train - predict(model, X_train)).^2));
            val_rmse(train_num) = rmse_val;
            
            % 選擇最佳模型
            if rmse_val < bestRMSE
                bestRMSE = rmse_val;  % 更新最佳RMSE
                bestModel = model;    % 保存最佳模型
            end
        end
        
        % 繪製訓練集和驗證集的RMSE變化
        figure;
        plot(1:numModels, train_rmse, '-o', 'DisplayName', '訓練集RMSE');
        hold on;
        plot(1:numModels, val_rmse, '-x', 'DisplayName', '驗證集RMSE');
        xlabel('模型反覆運算次數');
        ylabel('RMSE');
        title('訓練集與驗證集RMSE變化');
        legend('show');
        saveas(gcf, 'step2_訓練集與驗證集RMSE變化.png');
        
        %% 繪製最佳模型的預測與實際值對比
        Y_best_pred = predict(bestModel, X_val);  % 使用最佳模型進行預測
        Y_best_pred_int = round(Y_best_pred);  % 四捨五入
        
        % 計算並標註RMSE
        mse_best = mean((Y_val - Y_best_pred_int).^2);
        rmse_best = sqrt(mse_best);
        
        figure;
        scatter(Y_val, Y_best_pred_int, 'filled');
        hold on;
        plot([min(Y_val), max(Y_val)], [min(Y_val), max(Y_val)], 'r--'); % 理想的預測線
        xlabel('實際獎牌情況');
        ylabel('預測獎牌情況');
        title(['驗證集：最佳模型的預測 vs 實際值 (RMSE = ', num2str(rmse_best), ')']);
        text(min(Y_val), max(Y_best_pred_int), ['RMSE: ', num2str(rmse_best)], 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'left');
        grid on;
        saveas(gcf, 'step2_驗證集：最佳模型的預測 vs 實際值.png');
        
        %% 計算殘差
        residuals = Y_val - Y_best_pred_int;  % 計算殘差
        
        % 繪製殘差圖
        figure;
        plot(Y_val, residuals, 'o', 'MarkerFaceColor', 'b');
        hold on;
        yline(0, 'r--', 'LineWidth', 2); % y=0的輔助線
        xlabel('實際獎牌情況');
        ylabel('殘差');
        title(['驗證集：殘差圖 (RMSE = ', num2str(rmse_best), ')']);
        grid on;
        saveas(gcf, 'step2_驗證集：殘差圖.png');
        
        %% 繪製殘差分佈圖（長條圖）
        figure;
        histogram(residuals, 'Normalization', 'pdf', 'EdgeColor', 'black');
        hold on;
        % 繪製標準正態分佈曲線作為參考
        x = linspace(min(residuals), max(residuals), 100);
        y = normpdf(x, mean(residuals), std(residuals)); % 計算標準正態分佈
        plot(x, y, 'r-', 'LineWidth', 2);
        xlabel('殘差');
        ylabel('概率密度');
        title('驗證集：殘差分佈圖與標準正態分佈');
        grid on;
        saveas(gcf, 'step2_驗證集：殘差分佈圖與標準正態分佈.png');
        
        %% 使用最佳模型在測試集上測試
        Y_test_pred = predict(bestModel, X_test_scaled);  % 測試集預測
        Y_test_pred_int = round(Y_test_pred);  % 四捨五入
        
        % 計算信賴區間
        [Y_test_pred, Y_test_pred_ci] = predict(bestModel, X_test_scaled, 'Prediction', 'curve');
        Y_test_pred_lower = Y_test_pred_ci(:, 1); % 信賴區間下限
        Y_test_pred_upper = Y_test_pred_ci(:, 2); % 信賴區間上限
        
        % 生成列名：運動名稱 + 獎牌類型
        columnName = [char(fileName), '_', char(medaltype)];

        % 生成對應的信賴區間上下限的列名
        lowerColumnName = [columnName, '_Lower'];  % 例如 "Basketball_Gold_Lower"
        upperColumnName = [columnName, '_Upper'];  % 例如 "Basketball_Gold_Upper"

        % 將測試集的預測結果 (整數) 存入結果表格中
        % 這是四捨五入後的預測獎牌數
        resultsTable.(columnName) = Y_test_pred_int;

        % 將信賴區間的下限和上限分別存入 confidenceIntervals 表格
        % 這有助於評估預測的可信程度
        confidenceIntervals.(lowerColumnName) = Y_test_pred_lower; % 信賴區間下限
        confidenceIntervals.(upperColumnName) = Y_test_pred_upper; % 信賴區間上限
        
    end
end
