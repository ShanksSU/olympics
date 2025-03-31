clear; % 清除所有變數
clc; % 清除命令視窗
warning('off'); % 關閉警告
rng(123); % 設定隨機種子，以確保結果可重現

%% 讀取需要預測的大類
fileName = 'step1_output_PredictionData.csv'; % csv 檔路徑
data = readtable(fileName); % 讀取整個表格

% 提取第一列資料
firstColumn = data{:, 1};

% 去除重複值
uniqueList = unique(firstColumn); % 去除重複值並排序
numFiles = length(uniqueList);

% 獎牌類別
typelist = "Gold";

resultsTable = table();

weightTable = table();

% 初始化信賴區間表
confidenceIntervals = table();

% 設置信賴水準
alpha = 0.05; % 95% 信賴區間

totalTasks = 1 * 1;

% 當前任務計數器
currentTask = 0;

%% 迴圈處理每個檔
for fileIdx = 5:5

    for i = 1:1

        rng(123);

        currentTask = currentTask + 1; % 更新任務計數器
        progress = (currentTask / totalTasks) * 100; % 計算進度百分比

        % 列印進度資訊
        fprintf('正在處理文件: %s, 獎牌類型: %s, 進度: %.2f%%\n', ...
                cell2mat(uniqueList(fileIdx)), typelist, progress);
        
        medaltype = typelist; % 獲取當前元素

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

        % LASSO路徑圖
        lassoPlot(B, FitInfo, 'PlotType', 'Lambda');
        title('LASSO特徵重要性變化路徑圖');
        xlabel('Lambda');
        ylabel('特徵係數');
        saveas(gcf, 'step3_LASSO特徵重要性變化路徑圖.png');

        % 繪製交叉驗證誤差圖
        figure;
        plot(log(FitInfo.Lambda), FitInfo.MSE, 'LineWidth', 2);
        hold on;
        plot(log(Lambda1SE), FitInfo.MSE(FitInfo.Index1SE), 'ro', 'MarkerFaceColor', 'r');
        plot(log(LambdaMinMSE), FitInfo.MSE(FitInfo.IndexMinMSE), 'bo', 'MarkerFaceColor', 'b');
        xlabel('log(\lambda)');
        ylabel('均方誤差 (MSE)');
        title('交叉驗證誤差與 Lambda 的關係');
        legend('交叉驗證誤差', '最小均方誤差 + 1個標準差', '最小均方誤差');
        grid on;
        saveas(gcf, 'step3_LASSO 交叉驗證誤差.png');

        % 繪製最優Lambda下的係數圖
        figure;
        stem(B(:, FitInfo.Index1SE), 'LineWidth', 2);
        title(['最優Lambda = ', num2str(Lambda1SE), ' 下的特徵係數']);
        xlabel('特徵');
        ylabel('係數');
        grid on;
        saveas(gcf, 'step3_LASSO 最優Lambda下的特徵係數.png');

    end
end

