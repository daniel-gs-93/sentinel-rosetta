# Sentinel Rosetta Fine-tuning: Testing Document

## 1. Scope and Environment

- **Design Document：** `docs/DesignDocument_Finetune.md`。
- **In Scope：** 記錄欄位、`load_all` 契約、參考詞彙表與 CLI 參數（由 ADR 與測試本身定義）；3.1 節與 3.2 節的流程；3.3 節 adapter 的前提檢查與一次真實訓練；3.4 節的指令與接入翻譯服務的步驟；`ADR-0001`、`ADR-0002`、`ADR-0005`、`ADR-0006`、`ADR-0007`、`ADR-0008`、`ADR-0009`、`ADR-0011`、`ADR-0012` 的可觀察行為（詞條比對、訓練契約建構、留出切分與評分的規則由這些 ADR 定義）。
- **Out of Scope：** 真實詞彙表檔案（只在本機，不隨倉庫發布，沒有以它為輸入的測試，格式只以 `tmp_path` 內的檔案驗證）；評估分數與命中數的高低（只驗證流程與輸出格式，不驗證模型變好）；教師譯文是否正確、內容是否符合產製規則（沒有人工審閱；檢查指令只驗證欄位格式）；目標主機（Windows、CPU-only、4GB RAM）上的訓練與記憶體用量（未量測）。

## 2. Test Strategy

### 2.1 Tiers

| Tier | Boundary Under Test | Real Components | Replaced by Fake or Mock |
|---|---|---|---|
| Unit | Domain 純函式（詞條比對、切分、契約建構、評分）、兩個 repository、兩個 use case、`FinetuneRequest`、兩個 CLI、`HfSeq2SeqTrainer` 與 `HfPairTranslator` 的前提檢查 | 純邏輯；兩個 repository 讀 `tmp_path` 下的真實檔案 | `ScriptConverter`（fake）、`GeneratedRecordRepository`（fake）、`GlossaryRepository`（fake）、`ModelTrainer`（spy）、`PairTranslator`（fake）、use case（stub，於 CLI 測試）、套件檢查函式（fake，建構子注入） |
| Integration | 真實生成資料；兩個 repository 與 `OpenCcAdapter` 經 use case 組成的管線；`validate_generated_records_cli` 搭配真實 bootstrap | 資料檔、詞彙表檔、兩個 repository、`OpenCcAdapter`、兩個 use case | `ModelTrainer`（spy）、`PairTranslator`（fake），避免在此層載入模型 |
| Environment | `HfSeq2SeqTrainer`、`HfPairTranslator` 與 `finetune_cli` 對真實檢查點的訓練與翻譯 | 完整系統：`torch`、`accelerate`、`transformers`、真實檢查點 | 無 |

### 2.2 Partitions and Coverage

| Input | Technique | Partition or Boundary | Covered By |
|---|---|---|---|
| 記錄筆數（`build_training_pairs`） | Boundary Value Analysis | 0 筆、1 筆、多筆 | UT-18、UT-13、UT-17 |
| `source_lang` 與 `target_lang` | Equivalence Partitioning | `en→zh`、`zh→en`、兩者相同 | UT-13、UT-14、UT-19 |
| 文字內的空白 | Equivalence Partitioning | 頭尾空白、連續空格、換行、tab、全空白 | UT-15、UT-16 |
| 資料目錄中的檔案 | Equivalence Partitioning | 沒有 `*.jsonl`、1 個、多個、混有非 `*.jsonl` | UT-44、UT-41 至 UT-43 |
| 資料目錄本身 | Equivalence Partitioning | 存在、不存在 | UT-41、UT-44 |
| 方向資料夾（微調） | Equivalence Partitioning | 有記錄、不存在 | ET-02、ET-06 |
| 方向資料夾數（格式檢查） | Boundary Value Analysis | 0 個（根目錄不存在）、1 個、2 個 | IT-10、IT-07、IT-09 |
| 資料檔的一行 | Equivalence Partitioning | 合法、空白行、非 JSON、非物件、缺欄位、空字串欄位、表外欄位（含 `model`）、型別錯、`origin` 不符 | UT-41、UT-45 至 UT-47 |
| `id` 重複 | Equivalence Partitioning | 沒有、同一檔案、跨檔案 | UT-41、UT-48、UT-49 |
| 不同檔案的 `en` 相同 | Equivalence Partitioning | 相同（載入允許） | UT-50 |
| 詞彙表檔案 | Equivalence Partitioning | 合法、`abbreviation` 為 `null`、不存在、非 JSON、缺 `entries`、某筆不合規 | UT-51 至 UT-54 |
| 英文比對：縮寫 | Boundary Value Analysis | 完整 token、小寫、複數、嵌在較長 token 內（連字號、前綴字母、底線） | UT-01 至 UT-04 |
| 英文比對：全名 | Boundary Value Analysis | 多個單字（不分大小寫、複數）、單一單字、沒有縮寫的詞條 | UT-05 至 UT-07 |
| 中文比對 | Equivalence Partitioning | 子字串、`zh` 含全形括號 | UT-08、UT-09 |
| 比對的輸入與輸出 | Boundary Value Analysis | 空文字、空詞條、同一詞條出現多次 | UT-10、UT-11 |
| `holdout_fraction` | Boundary Value Analysis | 小於 0、0、正常值、剛好小於 1、不小於 1 | UT-21、UT-73、UT-74 |
| 留出筆數的取整 | Boundary Value Analysis | 剛好 .5（向上）、結果為 0、幾乎全部留出 | UT-22、UT-61 |
| chrF 的輸入 | Equivalence Partitioning | 相同、沒有共同字元、只差空白、hypothesis 為空、數量不同、多個句子 | UT-27 至 UT-33 |
| 預期詞條 | Equivalence Partitioning | 兩側都出現、reference 沒有（教師偏離）、沒有詞條 | UT-36、UT-38、UT-40 |
| 檢查點輸出的命中 | Equivalence Partitioning | 原始未命中而微調後命中、兩個翻譯方向 | UT-36、UT-37 |
| 輸出目錄 | Equivalence Partitioning | 不存在、存在且為空、存在且非空 | UT-85、UT-86、ET-01 |
| 訓練套件 | Equivalence Partitioning | 兩者都有、缺 `torch`、缺 `accelerate`、兩者都缺 | UT-87 |
| 基礎檢查點目錄 | Equivalence Partitioning | 完整、缺 `config.json`、缺權重檔 | UT-88、ET-01 |
| 評估用的檢查點目錄 | Equivalence Partitioning | 完整、缺 `config.json`、缺權重檔、缺 `torch` | UT-89、UT-90、ET-03 |
| 接入翻譯服務的檢查點 | Equivalence Partitioning | 完整的微調檢查點、不完整的檢查點 | ET-05 |
| CLI 結束碼 | State Transition | 成功 0、領域錯誤 1、驗證錯誤 1、缺參數 2 | UT-75、UT-77 至 UT-80、UT-83、UT-84、ET-06 |
| 儲存途中中斷 | None | — | None（見 Not Covered） |

## 3. Test Matrix

### 3.1 Unit

| Test ID | Claim | Technique | Procedure | Expected Result | Not Covered |
|---|---|---|---|---|---|
| UT-01 | ADR-0012 | Equivalence Partitioning | 文字 `The IFV arrived.`，詞條縮寫 `IFV`；`test_terms_in_text_english_abbreviation_matches_whole_token` | 回傳該詞條 | None |
| UT-02 | ADR-0012 | Equivalence Partitioning | 文字 `The ifv arrived.`；`test_terms_in_text_english_abbreviation_is_case_sensitive` | `()` | None |
| UT-03 | ADR-0012 | Equivalence Partitioning | 文字含 `IFVs`；`test_terms_in_text_english_abbreviation_allows_plural_suffix` | 回傳 `IFV` 詞條 | None |
| UT-04 | ADR-0012 | Boundary Value Analysis | 縮寫 `CA`，文字為 `CA-BDE`、`XCA`、`CA_1`（ids：`hyphen`、`letter-prefix`、`underscore`）；`test_terms_in_text_english_abbreviation_inside_longer_token_does_not_match` | 三個案例皆為 `()` | None |
| UT-05 | ADR-0012 | Boundary Value Analysis | 全名 `infantry fighting vehicle`，文字為 `Infantry Fighting Vehicles`；`test_terms_in_text_english_multi_word_full_name_matches_ignoring_case` | 回傳該詞條 | None |
| UT-06 | ADR-0012 | Boundary Value Analysis | 詞條縮寫 `AP`、全名 `transport`，文字 `The transport arrived.`；`test_terms_in_text_english_single_word_full_name_does_not_match` | `()` | None |
| UT-07 | ADR-0012 | Equivalence Partitioning | 縮寫為 `None`、全名 `course of action`；`test_terms_in_text_entry_without_abbreviation_matches_by_full_name` | 文字含全名時回傳該詞條 | None |
| UT-08 | ADR-0012 | Equivalence Partitioning | 文字含 `步兵战车`，詞條 `zh` 為 `步兵战车`；`test_terms_in_text_mandarin_matches_expected_zh_as_substring` | 回傳該詞條 | None |
| UT-09 | ADR-0012 | Equivalence Partitioning | 詞條 `zh` 為 `防空（高射）`，文字含 `防空` 但不含括號；`test_terms_in_text_mandarin_cuts_expected_zh_at_first_parenthesis` | 回傳該詞條 | None |
| UT-10 | ADR-0012 | Boundary Value Analysis | 空文字，以及空詞條 tuple（ids：`empty-text`、`empty-entries`）；`test_terms_in_text_empty_input_returns_empty_tuple` | 兩個案例皆為 `()` | None |
| UT-11 | ADR-0012 | Equivalence Partitioning | 文字先出現 `APC`，再出現兩次 `IFV` 與一次 `infantry fighting vehicle`，詞條順序為 `IFV`、`APC`；`test_terms_in_text_returns_each_entry_once_in_entries_order` | `(IFV 詞條, APC 詞條)` | None |
| UT-12 | ADR-0012 | Equivalence Partitioning | fake converter 把 `戰` 換成 `战`；`test_simplify_entries_changes_only_the_zh_field` | 每個詞條的 `zh` 被轉換，`abbreviation` 與 `full_name` 不變，順序不變 | 不驗證 OpenCC 的實際轉換（見 IT-03） |
| UT-13 | ADR-0007 第 2、3 項 | Equivalence Partitioning | `en→zh`，fake converter 把 `繁` 換成 `简`；`test_build_pairs_en_to_zh_source_is_english_target_is_simplified` | source 為英文，target 為轉換後的中文 | 不驗證 OpenCC 的實際轉換（見 IT-03） |
| UT-14 | ADR-0007 第 2、3 項 | Equivalence Partitioning | `zh→en` 同上；`test_build_pairs_zh_to_en_source_is_simplified_target_is_english` | source 為轉換後的中文，target 為英文 | 同 UT-13 |
| UT-15 | ADR-0007 第 1 項 | Equivalence Partitioning | 以頭尾空白、連續空格、換行、tab 的輸入參數化；`test_build_pairs_whitespace_runs_collapse_to_single_space` | 兩側皆為單一空格分隔、無頭尾空白 | None |
| UT-16 | ADR-0007 第 4 項 | Equivalence Partitioning | 任一側為空字串或全空白的記錄（ids：`empty-en`、`empty-zh`、`whitespace-en`、`whitespace-zh`）；`test_build_pairs_blank_side_drops_the_record` | 該筆不在結果中 | None |
| UT-17 | ADR-0007（順序、不去重） | Equivalence Partitioning | 輸入含重複內容的記錄；`test_build_pairs_keeps_duplicate_records_in_input_order` | 輸出順序與輸入相同，重複記錄各出現一次 | None |
| UT-18 | ADR-0007 第 4 項 | Boundary Value Analysis | 輸入空 tuple；`test_build_pairs_empty_records_returns_empty_tuple` | `()` | None |
| UT-19 | ADR-0007 第 3 項 | Equivalence Partitioning | 兩個方向相同（ids：`both-english`、`both-mandarin`）；`test_build_pairs_identical_languages_raises_value_error` | 拋出 `ValueError` | None |
| UT-20 | ADR-0007 第 1 項先於第 2 項 | Equivalence Partitioning | 以 spy converter 記錄參數；`test_build_pairs_converter_receives_normalized_text` | converter 收到的是空白正規化後的字串，且每筆只呼叫一次 | None |
| UT-21 | ADR-0009 | Boundary Value Analysis | `holdout_fraction` 為 0；`test_split_zero_fraction_keeps_every_record_for_training` | 留出集為空，訓練集等於輸入 | None |
| UT-22 | ADR-0009 | Boundary Value Analysis | 以（記錄數，比例）為（40, 0.25）、（10, 0.25）、（3, 0.5）、（1, 0.25）、（2, 0.99）參數化；`test_split_holdout_size_rounds_half_up` | 留出筆數依序為 10、3、2、0、2 | None |
| UT-23 | ADR-0009 要求 2 | Equivalence Partitioning | 同一批記錄以不同順序切分；`test_split_result_is_independent_of_record_order` | 兩次的留出 `id` 集合相同 | None |
| UT-24 | ADR-0009 | Equivalence Partitioning | 檢查切分的兩部分；`test_split_partitions_records_without_loss_or_overlap` | 兩部分的聯集等於輸入，交集為空，各自保持輸入的相對順序 | None |
| UT-25 | ADR-0009 要求 1 | Equivalence Partitioning | 同一批記錄用兩個 `seed` 切分；`test_split_different_seeds_select_different_holdouts` | 兩個留出集不同 | 不保證任兩個 `seed` 的結果都不同 |
| UT-26 | ADR-0009、ADR-0011 | Equivalence Partitioning | 兩個 `origin` 的混合記錄；測試內以 `sha256(f"{seed}:{id}")` 排序算出預期的留出 `id`；`test_split_holdout_selects_by_id_hash_regardless_of_origin` | 留出集的 `id` 等於測試算出的預期集合 | None |
| UT-27 | ADR-0008 | Equivalence Partitioning | 兩個相同字串；`test_chrf_score_identical_texts_returns_100` | 100.0 | None |
| UT-28 | ADR-0008 | Equivalence Partitioning | 兩個沒有共同字元的字串；`test_chrf_score_no_shared_characters_returns_zero` | 0.0 | None |
| UT-29 | ADR-0008 | Equivalence Partitioning | 以 4 組中英文字串對參數化（ids：`chinese`、`english`、`short-text`、`different-lengths`），期望值取自 `sacrebleu` 2.6.0 的 `CHRF().corpus_score`；`test_chrf_score_matches_sacrebleu_reference_values` | 與參考值的差距小於 1e-6 | 只涵蓋 4 組案例，沒有更大範圍的比對 |
| UT-30 | ADR-0008 | Equivalence Partitioning | hypothesis 與 reference 只差空白；`test_chrf_score_ignores_whitespace_differences` | 100.0 | None |
| UT-31 | ADR-0008 | Equivalence Partitioning | 兩個句對的整體分數，期望值取自 `sacrebleu` 的 `corpus_score`；`test_chrf_score_sums_statistics_across_sentences` | 與參考值的差距小於 1e-6，且不等於兩個句子分數的平均 | None |
| UT-32 | ADR-0008 | Boundary Value Analysis | hypothesis 為空字串；`test_chrf_score_empty_hypothesis_returns_zero` | 0.0 | None |
| UT-33 | ADR-0008 | Boundary Value Analysis | hypothesis 與 reference 數量不同；`test_chrf_score_length_mismatch_raises_value_error` | 拋出 `ValueError` | None |
| UT-34 | ADR-0008；Design 3.1 #5 | Equivalence Partitioning | 兩個句對與預先寫好的輸出，詞條為空；`test_build_report_scores_follow_chrf_score_function` | `sample_count` 為 2，每列含來源、參考答案、兩個輸出與各自 chrF，整體 chrF 等於對全部句子呼叫 `chrf_score` | None |
| UT-35 | ADR-0008 | Boundary Value Analysis | 輸出數量與句對數量不同；`test_build_report_length_mismatch_raises_value_error` | 拋出 `ValueError` | None |
| UT-36 | ADR-0012 | Equivalence Partitioning | `en→zh`，source 含 `IFV`，reference 含 `步兵战车`，詞條已是簡體，原始輸出不含、微調後輸出含；`test_build_report_en_to_zh_counts_terms_per_row` | 該列 `expected_term_count` 為 1，`base_hit_count` 為 0，`tuned_hit_count` 為 1 | None |
| UT-37 | ADR-0012 | Equivalence Partitioning | `zh→en`，source 含 `步兵战车`，reference 含 `IFV`，微調後輸出含 `IFV`、原始輸出不含；`test_build_report_zh_to_en_counts_terms_found_in_english_outputs` | 該列三個計數為 1、0、1 | None |
| UT-38 | ADR-0012 | Equivalence Partitioning | source 含 `U.S.`，reference 不含 `美國`（教師譯成 `美軍`）；`test_build_report_term_missing_from_reference_is_not_expected` | `expected_term_count` 為 0，兩個命中數為 0 | None |
| UT-39 | ADR-0008 | Equivalence Partitioning | 3 個句對，各列計數為（1, 0, 1）、（2, 1, 2）、（0, 0, 0）；`test_build_report_term_counts_sum_over_rows` | 整體 `expected_term_count` 為 3，`base_hit_count` 為 1，`tuned_hit_count` 為 3 | None |
| UT-40 | ADR-0008 | Boundary Value Analysis | 詞條為空 tuple；`test_build_report_without_entries_has_zero_term_counts` | 所有 `expected_term_count`、`base_hit_count`、`tuned_hit_count` 為 0 | None |
| UT-41 | ADR-0001、ADR-0011 | Equivalence Partitioning | 在 `tmp_path` 寫入一個含兩筆合法記錄的 `llm_generated.jsonl`；`test_load_all_single_file_returns_records_in_file_order` | 兩筆 `GeneratedRecord`，欄位與順序同檔案，中文未被改動，`origin` 為 `llm_generated` | None |
| UT-42 | ADR-0011（依檔名排序載入） | Equivalence Partitioning | 兩個檔案 `b_generated.jsonl` 與 `a_generated.jsonl`，各 2 筆；`test_load_all_two_files_returns_records_ordered_by_file_name_then_line` | 順序為 `a_generated` 的 2 筆，再 `b_generated` 的 2 筆 | None |
| UT-43 | ADR-0011（只載入 `*.jsonl`） | Equivalence Partitioning | 目錄中另有 `README.md` 與 `notes.txt`；`test_load_all_ignores_files_without_jsonl_extension` | 只回傳 `*.jsonl` 的記錄 | None |
| UT-44 | ADR-0011（沒有檔案時回傳空 tuple） | Boundary Value Analysis | 空目錄與不存在的目錄（ids：`empty-dir`、`missing-dir`）；`test_load_all_directory_without_jsonl_files_returns_empty_tuple` | 兩個案例皆為 `()` | None |
| UT-45 | ADR-0011（略過空白行） | Equivalence Partitioning | 檔案含空白行；`test_load_all_blank_lines_are_skipped` | 只回傳有內容的記錄 | None |
| UT-46 | ADR-0001 | Equivalence Partitioning | 第 2 行不是 JSON；`test_load_all_invalid_json_error_locates_the_failing_line` | 拋出 `GeneratedRecordFormatError`，訊息含檔案路徑與 `2` | None |
| UT-47 | ADR-0001、ADR-0011 | Equivalence Partitioning | 以 8 種不合法記錄參數化（ids：`not-an-object`、`missing-field`、`blank-zh`、`blank-en`、`unknown-field`、`model-field`、`non-string-id`、`origin-mismatch`）；`test_load_all_invalid_record_raises_generated_record_format_error` | 每種皆拋出 `GeneratedRecordFormatError` | 只驗證錯誤型別，不逐一比對訊息文字 |
| UT-48 | ADR-0005 | Equivalence Partitioning | 同一檔案兩筆相同 `id`；`test_load_all_duplicate_id_in_one_file_raises_generated_record_format_error` | 拋出 `GeneratedRecordFormatError` | None |
| UT-49 | ADR-0005、ADR-0011 | Equivalence Partitioning | 兩個檔案共用一個 `id`；`test_load_all_duplicate_id_across_files_raises_error_naming_both_files` | 拋出 `GeneratedRecordFormatError`，訊息含兩個檔案名 | None |
| UT-50 | ADR-0011（`en` 重複不是載入錯誤） | Equivalence Partitioning | 兩個檔案有相同的 `en`、不同的 `id`；`test_load_all_same_en_in_two_files_is_not_an_error` | 兩筆都載入 | None |
| UT-51 | ADR-0012 | Equivalence Partitioning | 在 `tmp_path` 寫入含 `meta` 與 2 個詞條（其中一個有 `note`）的檔案；`test_load_valid_glossary_returns_entries_in_file_order` | 2 個 `GlossaryEntry`，三個欄位等於檔案內容，順序同檔案 | None |
| UT-52 | `abbreviation` 可為 `null` | Equivalence Partitioning | 詞條的 `abbreviation` 為 `null`；`test_load_glossary_null_abbreviation_becomes_none` | `GlossaryEntry.abbreviation` 為 `None` | None |
| UT-53 | 詞彙表不合規 → `GlossaryFormatError` | Equivalence Partitioning | 以 7 種不合法檔案參數化（ids：`missing-file`、`not-json`、`missing-entries`、`entry-not-object`、`blank-full-name`、`blank-zh`、`abbreviation-wrong-type`）；`test_load_invalid_glossary_raises_glossary_format_error` | 每種皆拋出 `GlossaryFormatError` | None |
| UT-54 | 詞彙表錯誤訊息含路徑與序號 | Equivalence Partitioning | 第 2 個詞條缺 `zh`；`test_load_invalid_entry_error_locates_the_entry` | 訊息含檔案路徑與 `2` | None |
| UT-55 | Design 3.1 #1；ADR-0011 | Equivalence Partitioning | fake repository 回傳 3 筆記錄，記錄 `load_all` 的呼叫次數；`test_execute_loads_every_record_once` | `load_all` 被呼叫 1 次，沒有任何依來源過濾 | None |
| UT-56 | Design 3.1 #4 | Equivalence Partitioning | spy trainer 記錄參數；`test_execute_delegates_training_to_the_trainer` | trainer 收到預期的 pair tuple 與同一個 config 物件 | None |
| UT-57 | Design 3.1 #4、#6 | Equivalence Partitioning | trainer 回傳固定結果；`test_execute_returns_trainer_result` | use case 回傳該結果 | None |
| UT-58 | Design 3.1 #3 | Equivalence Partitioning | repository 回傳空 tuple；`test_execute_no_pairs_raises_no_training_data_error` | 拋出 `NoTrainingDataError`，trainer 未被呼叫 | None |
| UT-59 | Design 3.1 #1；ADR-0009 | Equivalence Partitioning | 8 筆記錄、比例 0.25，spy trainer；`test_execute_holdout_excludes_heldout_records_from_training` | trainer 收到 6 個 pair，且不含留出記錄的 source | None |
| UT-60 | Design 3.1 #5 至 #6 | Equivalence Partitioning | fake translator 回傳預先寫好的輸出，fake repository 提供含 `IFV` 的詞條；`test_execute_holdout_attaches_report_comparing_both_checkpoints` | `TrainingResult.evaluation` 含 2 個留出句、兩個檢查點的輸出、各自 chrF 與詞條計數 | translator 為 fake，不驗證真實翻譯 |
| UT-61 | Design 3.1 #3 | Boundary Value Analysis | 1 筆記錄、比例 0.25（留出 0 筆）；`test_execute_holdout_too_small_raises_error_before_training` | 拋出 `NoEvaluationDataError`，trainer 與 translator 皆未被呼叫 | None |
| UT-62 | Design 3.1 #1 至 #2、#6 | Equivalence Partitioning | 比例為 0；`test_execute_without_holdout_skips_evaluation` | translator 與詞彙表 repository 未被呼叫，`evaluation` 為 `None` | None |
| UT-63 | Design 3.1 #2 | Equivalence Partitioning | 以呼叫順序紀錄 glossary `load` 與 trainer `train`；`test_execute_holdout_loads_glossary_before_training` | `load` 早於 `train` | None |
| UT-64 | Design 3.1 #2 | Equivalence Partitioning | fake glossary repository 拋出 `GlossaryFormatError`；`test_execute_holdout_glossary_error_propagates_before_training` | 同一例外往上拋，trainer 未被呼叫 | None |
| UT-65 | Design 3.1 #2 | Equivalence Partitioning | fake converter 把 `戰` 換成 `战`，詞條 `zh` 為 `步兵戰車`，reference 與 tuned 輸出含 `步兵战车`；`test_execute_holdout_matches_terms_against_simplified_entries` | 該列 `expected_term_count` 與 `tuned_hit_count` 皆為 1 | None |
| UT-66 | Design 3.2 #2 | Equivalence Partitioning | fake repository 提供 3 筆記錄；`test_execute_returns_the_number_of_loaded_records` | 回傳 3 | None |
| UT-67 | Design 3.2 #2 | Boundary Value Analysis | fake repository 回傳空 tuple；`test_execute_no_records_returns_zero` | 回傳 0，不拋例外 | 空生成資料對微調是錯誤（`NoTrainingDataError`），對格式檢查不是 |
| UT-68 | Design 3.2 #1 | Equivalence Partitioning | fake record repository 拋出 `GeneratedRecordFormatError`；`test_execute_format_error_propagates` | 同一個例外往上拋 | None |
| UT-69 | CLI 參數的預設值 | Boundary Value Analysis | 只給兩個語言；`test_finetune_request_valid_arguments_use_documented_defaults` | `epochs` 3、`batch_size` 8、`learning_rate` 5e-5、`holdout_fraction` 0、`max_length` 128、`seed` 42 | None |
| UT-70 | 參數超出範圍 → `ValidationError` | Boundary Value Analysis | 以 `epochs`=0、`batch_size`=0、`learning_rate`=0 與負值、兩語言相同參數化（ids：`epochs-zero`、`batch-size-zero`、`learning-rate-zero`、`learning-rate-negative`、`same-language`）；`test_finetune_request_invalid_value_raises_validation_error` | 皆拋出 `ValidationError` | None |
| UT-71 | `--base-model-dir`、`--output-dir` 的預設路徑 | Equivalence Partitioning | 兩個方向（ids：`en-to-zh`、`zh-to-en`）；`test_finetune_request_resolved_dirs_follow_direction_defaults` | `models/hf/opus-mt-<s>-<t>` 與 `models/finetuned/opus-mt-<s>-<t>` | None |
| UT-72 | `--base-model-dir`、`--output-dir` 可覆寫預設 | Equivalence Partitioning | 明確指定兩個目錄；`test_finetune_request_explicit_dirs_override_defaults` | 回傳指定的路徑 | None |
| UT-73 | `--holdout-fraction` 範圍 0 以上、未滿 1 | Boundary Value Analysis | 以 -0.1、1.0、1.5 參數化（ids：`negative`、`one`、`above-one`）；`test_finetune_request_invalid_holdout_fraction_raises_validation_error` | 皆拋出 `ValidationError` | None |
| UT-74 | `--holdout-fraction` 範圍 0 以上、未滿 1 | Boundary Value Analysis | `holdout_fraction` 為 0.99；`test_finetune_request_holdout_fraction_just_below_one_accepts` | 建立成功 | None |
| UT-75 | Design 3.4 微調 `en→zh`，全部生成資料 | State Transition | stub use case 回傳結果；`test_main_valid_arguments_returns_zero_with_summary_on_stdout` | exit code 0，stdout 為 `fine-tuned <n> pairs, train_loss=<x>, saved to <dir>` | 不驗證真實訓練 |
| UT-76 | ADR-0007（CLI 只呼叫 use case） | Equivalence Partitioning | 帶全部旗標執行；`test_main_passes_parsed_arguments_to_use_case` | stub 收到解析後的語言、`holdout_fraction` 與 `TrainingConfig`；組裝函式收到來源語言、目標語言、兩個目錄與 `max_length` | None |
| UT-77 | `-s`、`-t` 必須不同 | State Transition | 兩語言相同；`test_main_source_equals_target_returns_one_with_message_on_stderr` | exit code 1，stderr 非空，use case 未被呼叫 | None |
| UT-78 | 領域例外 → CLI exit code 1 | State Transition | stub 拋出五種領域例外（ids：`no-training-data`、`record-format`、`glossary-format`、`no-evaluation-data`、`training-environment`）；`test_main_domain_failure_returns_one_with_message_on_stderr` | 每種皆 exit code 1，stderr 含例外訊息 | 未預期的例外不經 CLI 處理 |
| UT-79 | `-s`、`-t` 必填 | State Transition | 缺 `-t`；`test_main_missing_required_argument_exits_with_code_two` | 拋出 `SystemExit`，code 為 2 | None |
| UT-80 | ADR-0011 | State Transition | 傳入已移除的 `--include-machine-generated`；`test_main_removed_include_machine_generated_flag_exits_with_code_two` | 拋出 `SystemExit`，code 為 2 | None |
| UT-81 | Design 3.4 微調 `en→zh`，留出 25% 作評估 | State Transition | stub use case 回傳含報告的結果；`test_main_holdout_prints_evaluation_report` | stdout 第 2 行為 `held-out <m> pairs: chrF base=<a>, tuned=<b>`（分數取兩位小數），第 3 行為 `terms: base=<p>/<q>, tuned=<r>/<q>`，stderr 逐句列出參考答案與兩個翻譯 | None |
| UT-82 | Design 3.4 檢查資料檔的欄位格式 | State Transition | stub use case 回傳 3；`test_main_valid_records_returns_zero_with_summary_on_stdout` | exit code 0，stdout 為 `checked 3 records` | None |
| UT-83 | ADR-0011 | State Transition | stub 拋出 `GeneratedRecordFormatError`；`test_main_format_error_returns_one_with_message_on_stderr` | exit code 1，stderr 含例外訊息 | None |
| UT-84 | 格式檢查 CLI 沒有參數 | State Transition | 傳入未定義的 `--records-dir`；`test_main_unexpected_argument_exits_with_code_two` | 拋出 `SystemExit`，code 為 2 | None |
| UT-85 | Design 3.3 `Output dir free?`；ADR-0006 | Equivalence Partitioning | 輸出目錄非空；`test_train_output_dir_not_empty_raises_error_without_touching_files` | 拋出 `TrainingEnvironmentError`，既有檔案內容不變 | None |
| UT-86 | Design 3.3 `Output dir free?` | Boundary Value Analysis | 輸出目錄存在且為空，其餘前提缺失；`test_train_empty_output_dir_passes_output_check` | 錯誤來自套件檢查，不是輸出目錄 | None |
| UT-87 | Design 3.3 `Packages found?`；ADR-0006 | Equivalence Partitioning | monkeypatch `find_spec` 使 `torch`、`accelerate` 缺其一或皆缺（ids：`torch`、`accelerate`、`both`）；`test_train_missing_packages_error_names_each_missing_package` | 訊息含每個缺少的套件名 | 不驗證套件版本相容性 |
| UT-88 | Design 3.3 `Base model found?` | Equivalence Partitioning | 套件齊全，基礎目錄缺 `config.json` 或缺權重檔（ids：`nothing`、`config-only`、`weights-only`）；`test_train_incomplete_base_model_raises_error_with_directory` | 拋出 `TrainingEnvironmentError`，訊息含目錄路徑 | 不驗證權重檔內容是否損毀 |
| UT-89 | Design 3.1 #5；檢查點目錄不完整 → `TrainingEnvironmentError` | Equivalence Partitioning | 檢查點目錄缺 `config.json` 或權重檔（ids：`nothing`、`config-only`、`weights-only`）；`test_translate_incomplete_model_dir_raises_error_with_directory` | 拋出 `TrainingEnvironmentError`，訊息含目錄路徑 | 不驗證權重檔內容是否損毀 |
| UT-90 | Design 3.1 #5；缺 `torch` → `TrainingEnvironmentError` | Equivalence Partitioning | 套件檢查回報缺 `torch`；`test_translate_missing_torch_raises_training_environment_error` | 拋出 `TrainingEnvironmentError`，訊息含 `torch` | None |
| UT-91 | Design 3.2 #1 至 #2；ADR-0013 | Boundary Value Analysis | 兩個 fake repository 各回傳 3 筆與 2 筆記錄；`test_execute_two_repositories_returns_the_sum_of_their_records` | 回傳 5 | None |

### 3.2 Integration

| Test ID | Claim | Technique | Procedure | Expected Result | Not Covered |
|---|---|---|---|---|---|
| IT-01 | ADR-0001、ADR-0011 | Equivalence Partitioning | 以 `JsonlGeneratedRecordRepository` 載入 `data/llm_generated/en_to_zh/`；`test_generated_directory_parses_into_records` | 無 `GeneratedRecordFormatError`，至少一筆記錄，且每筆 `origin` 都是某個 `*.jsonl` 的主檔名 | 不驗證譯文是否正確；不寫死筆數 |
| IT-02 | ADR-0005 | Equivalence Partitioning | 檢查 `data/llm_generated/en_to_zh/` 所有記錄的 `id`；`test_generated_ids_are_unique_hex_strings` | 全為 32 字元小寫十六進位，且全部不重複 | None |
| IT-03 | ADR-0002、ADR-0007 第 2 項 | Equivalence Partitioning | 把一筆繁體記錄寫到 `tmp_path`，以真實 repository、`OpenCcAdapter` 與 spy trainer 執行 `en→zh`；`test_pipeline_traditional_records_reach_trainer_as_simplified_pairs` | trainer 收到唯一一個 pair：(`Their commercial space sector features partially state-owned enterprises.`, `其商业太空部门，包括遥感探测、发射与通信服务，具有部分国有的企业`) | trainer 為 spy，不驗證 tokenize；不使用 `data/llm_generated/` 的實際內容 |
| IT-04 | Design 3.1 #1；ADR-0011 | Equivalence Partitioning | `tmp_path` 內兩個檔案各 3 筆與 2 筆記錄，同一管線；`test_pipeline_every_jsonl_file_contributes_pairs_to_the_trainer` | trainer 收到的 pair 數為 5 | None |
| IT-05 | Design 3.1 #1 至 #3；ADR-0009 | Equivalence Partitioning | 8 筆繁體記錄寫到 `tmp_path`，真實 repository 與 `OpenCcAdapter`、spy trainer、fake translator，比例 0.25；`test_pipeline_holdout_sentences_never_reach_the_trainer` | 報告的 2 個留出 source 都不在 trainer 收到的 6 個 pair 中 | trainer 與 translator 皆為替身 |
| IT-06 | Design 3.1 #2、#5 至 #6；ADR-0012 | Equivalence Partitioning | 在 `tmp_path` 寫入生成資料、含 `IFV` 詞條的詞彙表檔，真實兩個 repository 與 `OpenCcAdapter`，fake translator 的微調後輸出含 `步兵战车`、原始輸出不含；`test_pipeline_holdout_report_counts_terms_from_the_glossary_file` | `expected_term_count` 與 `tuned_hit_count` 為 1，`base_hit_count` 為 0 | translator 為 fake |
| IT-07 | Design 3.4 檢查資料檔的欄位格式 | State Transition | monkeypatch `validate_generated_records_bootstrap` 的資料根目錄到 `tmp_path`，`tmp_path/en_to_zh/` 內檔案格式合規；`test_validate_cli_real_run_valid_records_returns_zero_with_stdout_summary` | exit code 0，stdout 為 `checked <n> records`，`<n>` 等於寫入的筆數 | None |
| IT-08 | Design 3.4 檢查資料檔的欄位格式 | State Transition | 同上，第 2 行缺少 `zh` 欄位；`test_validate_cli_real_run_malformed_records_returns_one_with_stderr_text` | exit code 1，stderr 含檔案名與 `2` | None |
| IT-09 | Design 3.4 檢查資料檔的欄位格式；ADR-0013 | Boundary Value Analysis | 在 `tmp_path` 建 `en_to_zh/`（2 筆）與 `zh_to_en/`（1 筆），monkeypatch `validate_generated_records_bootstrap` 的資料根目錄；`test_validate_cli_real_run_counts_records_of_every_direction_folder` | exit code 0，stdout 為 `checked 3 records` | 根目錄直下的 `*.jsonl` 不被檢查，這個行為沒有測試 |
| IT-10 | Design 3.4 檢查資料檔的欄位格式；ADR-0013 | Boundary Value Analysis | monkeypatch 資料根目錄到不存在的路徑；`test_validate_cli_real_run_missing_generated_root_reports_zero_records` | exit code 0，stdout 為 `checked 0 records` | None |

### 3.3 Environment

| Test ID | Claim | Technique | Procedure | Expected Result | Not Covered |
|---|---|---|---|---|---|
| ET-01 | Design 3.3 `Tokenize`、`Train`、`Save`；ADR-0006 | State Transition | 以真實檢查點與 3 個 pair 訓練 1 個 epoch 後重新載入輸出；`test_train_real_checkpoint_writes_reloadable_checkpoint` | `pair_count` 為 3，`train_loss` 為有限正數，輸出目錄含 `config.json` 與權重檔，且 `AutoModelForSeq2SeqLM` 與 `AutoTokenizer` 可載入 | 不驗證翻譯品質；不驗證 CPU-only 4GB RAM 主機上的行為 |
| ET-02 | Design 3.4 微調 `en→zh`，全部生成資料；Design 3.1 | State Transition | monkeypatch bootstrap 的資料根目錄到 `tmp_path`（`en_to_zh/` 內 8 筆記錄），在真實檢查點上執行 CLI，`--epochs 1`；`test_cli_real_run_trains_records_with_summary_on_stdout` | exit code 0，stdout 符合摘要格式且 `<n>` 為 8，輸出目錄含檢查點 | 同 ET-01 |
| ET-03 | Design 3.1 #5；ADR-0006 | State Transition | 以真實的原始檢查點翻譯 3 個英文句子；`test_translate_real_checkpoint_returns_one_output_per_source` | 回傳 3 個非空字串，順序與輸入一致 | 不驗證翻譯品質 |
| ET-04 | Design 3.4 微調 `en→zh`，留出 25% 作評估；Design 3.1 #1 至 #6 | State Transition | monkeypatch 資料根目錄與詞彙表路徑到 `tmp_path`（`en_to_zh/` 內 8 筆記錄、含 1 個詞條），以真實檢查點執行 CLI，`--holdout-fraction 0.25 --epochs 1`；`test_cli_real_run_with_holdout_prints_three_stdout_lines` | exit code 0；stdout 第 1 行的 pair 數為 6，第 2 行符合 `held-out 2 pairs: chrF base=<a>, tuned=<b>`，第 3 行符合 `terms: base=<p>/<q>, tuned=<r>/<q>` | 不驗證分數與命中數的高低 |
| ET-05 | Design 3.4 把微調後的檢查點接入翻譯服務；ADR-0010 | State Transition | 以 ET-01 的方式用 3 個 pair 訓練 1 個 epoch，在 `tmp_path` 以 `TransformersConverter` 轉成 `opus-mt-en-zh-ct2`（int8），把該檢查點的 tokenizer 檔案複製為 `opus-mt-en-zh`，並從專案的 `models/` 複製 `opus-mt-zh-en` 與 `opus-mt-zh-en-ct2`（`CTranslate2Adapter` 需要兩個方向）；以 `CTranslate2Adapter(models_dir=<tmp_path>)` 翻譯 `Hello`；`test_converted_checkpoint_is_translatable_by_the_engine` | 轉換產生 `config.json`、`model.bin`、`shared_vocabulary.json`；`translate()` 回傳非空的 `target_text` | 不驗證翻譯品質；不驗證 `README.md` 的指令字串，只驗證同一組 API 呼叫；檢查點不完整時的轉換失敗是函式庫行為，沒有測試 |
| ET-06 | Design 3.4 微調 `zh→en`；ADR-0013 | State Transition | monkeypatch 資料根目錄到 `tmp_path`，只有 `en_to_zh/`（8 筆記錄），執行 `-s zh -t en --epochs 1`；`test_cli_real_run_missing_direction_folder_returns_one_with_stderr_text` | exit code 1，stderr 以 `No training pairs remain` 開頭，輸出目錄不存在 | 同 ET-01 |
