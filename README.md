# Grounhug's Geolocation research

## 如何開始
先將原始資料檔放在`data/`資料夾，並執行`src/detect_move_static.py`。此腳本會解析原始資料，並分別分析每個`imsi`的enodeb連線模式，並分辨其移動/靜止的區間。分析結果會以視覺化圖片和csv檔的方式呈現。前者儲存在`img/`資料夾，呈現每個`imsi`的連線模式序列和移動/靜止。後者儲存在`result/outputcsv`，將移動/靜止狀態貼回原始資料集。此外`result/elimiated_imsi.csv`標記的是因為資料不足而被捨棄的`imsi`。

如果資料集太大(譬如jakarta_sample.csv)，可使用`src/data_sampling.py`抽取特定數量imsi的資料，另存成另一個資料小資料集。此程式會優先輸出資料點多的imsi，從最多的開始輸出

## 參數設定
1. VECTORIZE_METHOD: 設定使用哪個分析模式，設定為`enodeb_base`會採用向量化enodeb，並計算每個time frame的向量合的方式賦予time frame向量；設定為`time_frame_base`會由PCA直接向量化time frame。目前測試結果顯示`enodeb_base`效果較理想
2. START_TIME & END_TIME: 原始資料集的可用時間段，必須根據不同資料集設定不同數值。`150men.csv`可設為2022/9/17到2022/9/20. `jakarta_sample.csv`可設為2022/9/28到2022/10/1
3. RAW_DATA_FILE_NAME: 原始資料集名稱
4. TIME_FRAME_INTERVAL: time frame的長度，單位是秒
5. WINDOW_SIZE: 建構PCA所需要的enodeb co-occurrence array的時間窗口寬度，單位是秒。時間窗口是個滑動窗口，在窗口內同一個imsi連線的不同enodebs會被視為共同出現的emodebs，並作為記錄寫入array，作為PCA的input。
6. PCA_DIM: PCA的principle commonets數量

## `imsi`篩選條件
必須要在time frame有超過30個frame有資料才會被納入分析
