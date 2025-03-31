clear; % 清除工作區中的所有變數
clc; % 清除命令窗口的內容
rng(123); % 設置隨機數種子，確保結果的可重現性

%% 設定檔案路徑和輸出文件名
folderPath = '../dataset/clean/TrainData'; % 指定包含 CSV 文件的資料夾
outputFile = 'step1_output_PredictionData.csv'; % 設置輸出文件名

% 獲取資料夾內所有 CSV 文件的信息
files = dir(fullfile(folderPath, '*.csv')); % 獲取該目錄下所有 .csv 文件的信息
numFiles = length(files); % 計算文件總數

% 初始化存儲結果的表格，包含 10 個變數（列）
resultTable = table('Size', [0, 10], 'VariableTypes', {'string', 'double', 'double', 'double', 'double', ...
                                                        'double', 'double', 'double', 'double', 'double'}, ...
                     'VariableNames', {'FileName', 'Run', 'MSE_Train', 'RMSE_Train', 'R2_Train', 'MAE_Train', ...
                                       'MSE_Test', 'RMSE_Test', 'R2_Test', 'MAE_Test'});

%% 遍歷所有文件，逐一處理
for fileIdx = 1:numFiles
    fileName = files(fileIdx).name; % 取得當前文件名稱
    fprintf('正在處理文件: %s\n', fileName); % 在命令窗口輸出當前處理的文件名
    
    % 構建完整的文件路徑
    filePath = fullfile(folderPath, fileName);
    
    %% 加載數據
    summerOly_athletes = readtable(filePath); % 讀取 CSV 文件內容

    % 重新命名列名，以統一格式
    numColumns = size(summerOly_athletes, 2); % 獲取總列數
    newColumnNames = strings(1, numColumns); % 初始化新列名的字串數組
    for i = 1:numColumns
        newColumnNames(i) = "x" + i; % 生成列名 x1, x2, x3, ...
    end
    summerOly_athletes.Properties.VariableNames = newColumnNames; % 設置新列名
    
    %% 篩選目標數據（僅選擇 x1 值為 'All' 的行）
    filteredTable = summerOly_athletes(strcmp(summerOly_athletes.x1, 'All'), :);
    
    %% 提取特徵 (X) 和目標變數 (Y)
    X = filteredTable{:, 5:end}; % 特徵變數從第 5 列開始
    Y = filteredTable{:, 2}; % 目標變數 Y
    
    %% Z-score 標準化數據（均值為 0，標準差為 1）
    [X_scaled, mu, sigma] = zscore(X);
    
    for run = 1:20 % 執行 20 次模型訓練與測試
        
        %% 切分數據集 (67% 訓練, 33% 測試)
        cv = cvpartition(size(X_scaled, 1), 'HoldOut', 0.33); % 使用交叉驗證劃分數據
        X_train = X_scaled(training(cv), :);
        Y_train = Y(training(cv));
        X_test = X_scaled(test(cv), :);
        Y_test = Y(test(cv));
        
        %% 訓練線性回歸模型
        model = fitlm(X_train, Y_train); % 構建線性回歸模型
        
        % 預測訓練集和測試集結果
        Y_train_pred = predict(model, X_train);
        Y_test_pred = predict(model, X_test);
        
        % 將預測結果四捨五入為整數
        Y_train_pred_int = round(Y_train_pred);
        Y_test_pred_int = round(Y_test_pred);
        
        % 計算模型評估指標
        mse_train = mean((Y_train - Y_train_pred_int).^2); % 訓練集均方誤差（MSE）
        r2_train = 1 - sum((Y_train - Y_train_pred_int).^2) / sum((Y_train - mean(Y_train)).^2); % 訓練集 R2
        rmse_train = sqrt(mse_train); % 訓練集均方根誤差（RMSE）
        mae_train = mean(abs(Y_train - Y_train_pred_int)); % 訓練集平均絕對誤差（MAE）
        
        mse_test = mean((Y_test - Y_test_pred_int).^2); % 測試集 MSE
        r2_test = 1 - sum((Y_test - Y_test_pred_int).^2) / sum((Y_test - mean(Y_test)).^2); % 測試集 R2
        rmse_test = sqrt(mse_test); % 測試集 RMSE
        mae_test = mean(abs(Y_test - Y_test_pred_int)); % 測試集 MAE
        
        % 存儲當前結果
        newRow = {fileName, run, mse_train, rmse_train, r2_train, mae_train, ...
                  mse_test, rmse_test, r2_test, mae_test};
        resultTable = [resultTable; newRow];
    end
end

%% 篩選結果，剔除無效數據
% 只保留 mse_train 和 mse_test 都不為 0 的行
filteredTable = resultTable(resultTable.MSE_Train ~= 0 & resultTable.MSE_Test ~= 0, :);

% 統計每個 FileName 出現的次數
fileNameCounts = varfun(@numel, filteredTable, 'InputVariables', 'FileName', 'GroupingVariables', 'FileName');

% 只保留出現次數為 20 的 FileName
validFileNames = fileNameCounts(fileNameCounts.numel_FileName == 20, :).FileName;
finalTable = filteredTable(ismember(filteredTable.FileName, validFileNames), :);

%% 將結果保存至 CSV 文件
writetable(finalTable, outputFile);
fprintf('所有結果已保存到文件: %s\n', outputFile);