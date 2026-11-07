# Sentinel Rosetta: Testing Document

## 1. Scope and Environment

- **Design Document：** `docs/DesignDocument.md`。
- **In Scope：** 2.1 節的元件與 2.2 節的 interface；3.1 至 3.4 節的流程；3.5 節 `ClientUI` 的錯誤處理；3.6 節的指令；`ADR-0003`、`ADR-0004`、`ADR-0010` 的可觀察行為。
- **Out of Scope：** 專有名詞譯名是否出現（由權重決定，衡量方式見 `docs/TestDocument_Finetune.md` 的術語命中率）；翻譯品質；目標主機（Windows、CPU-only、4GB RAM）上的行為；3.5 節 `ClientUI` 的非 JSON、`ok` 為 false、缺 `target_text` 三個分支（沒有測試，人工難以重現，見 ET-04）。

## 2. Test Strategy

### 2.1 Tiers

| Tier | Boundary Under Test | Real Components | Replaced by Fake or Mock |
|---|---|---|---|
| Unit | `TranslateTextUseCase`、`TranslationRequest`、`WebController`、`CliController`、`OpenCcAdapter`、`_offline_env` | 純邏輯；`OpenCcAdapter` 呼叫真實 `opencc`（無外部資源） | `TranslationEngine`（fake）、`ScriptConverter`（spy）、`TranslateTextUseCase`（stub，於 controller 與 CLI 測試） |
| Integration | `CTranslate2Adapter` 與真實權重；`TranslateTextUseCase` 搭配真實 adapter | `CTranslate2Adapter`、`OpenCcAdapter`、`TranslateTextUseCase` | 無 |
| Environment | CLI 與 HTTP 入口在真實權重下的端到端行為；`ClientUI` | 完整系統 | 無 |

### 2.2 Partitions and Coverage

| Input | Technique | Partition or Boundary | Covered By |
|---|---|---|---|
| 翻譯方向 | Equivalence Partitioning | `en→zh`、`zh→en` | UT-01 至 UT-07、IT-04、IT-05 |
| `source_lang` 與 `target_lang` | Equivalence Partitioning | 不同、相同 | UT-09、UT-10、UT-14、UT-19 |
| `text` 字元數 | Boundary Value Analysis | 0（拒絕）、1（接受） | UT-11、UT-12 |
| `text` 的 token 數 | Boundary Value Analysis | 512（接受）、513（拒絕） | IT-02、IT-03 |
| 引擎例外 | Equivalence Partitioning | `TranslationInputTooLongError`、其他例外 | UT-08、UT-15、UT-16、UT-20 |
| 簡繁轉換的輸入 | Equivalence Partitioning | 繁體、簡體、空字串 | UT-21 至 UT-25 |
| `zh→en` 的輸入字形 | Equivalence Partitioning | 繁體 | IT-05 |
| 離線環境變數 | Equivalence Partitioning | 未設定、已設定 | UT-26、UT-27 |
| `ClientUI` 的回應 | State Transition | 連線失敗 | ET-04 |
| `ClientUI` 的回應 | State Transition | 成功 | ET-05 |
| `ClientUI` 的回應 | State Transition | 非 JSON、`ok` 為 false、缺 `target_text` | None |
| 模型權重的方向目錄 | Equivalence Partitioning | 兩個方向都存在、缺一個方向 | IT-01、IT-06 |
| 啟動時缺少權重的 `RuntimeError` | Equivalence Partitioning | CLI、API 伺服器 | UT-29、UT-30 |

## 3. Test Matrix

### 3.1 Unit

| Test ID | Claim | Technique | Procedure | Expected Result | Not Covered |
|---|---|---|---|---|---|
| UT-01 | Design 3.3 #1；ADR-0010 | Equivalence Partitioning | fake engine 記錄收到的文字，`en→zh` 執行；`test_execute_en_to_zh_sends_original_text_to_engine` | engine 收到的文字等於輸入原文 | 不驗證真實 tokenizer 的處理 |
| UT-02 | Design 3.3 #2 | Equivalence Partitioning | engine 回傳簡體文字，converter 為 spy；`test_execute_en_to_zh_converts_engine_output_to_traditional` | `to_traditional` 收到 engine 的輸出，`target_text` 等於 converter 的回傳值 | 不驗證 OpenCC 的實際轉換（見 IT-04） |
| UT-03 | Design 3.3 #1 至 #2 | Equivalence Partitioning | 記錄 engine 與 converter 的呼叫順序；`test_execute_en_to_zh_converts_script_only_after_translating` | 呼叫順序為 `translate`、`to_traditional`，`to_simplified` 沒有被呼叫 | None |
| UT-04 | Design 3.4 #1；ADR-0010 | Equivalence Partitioning | converter 的 `to_simplified` 回傳固定值，`zh→en` 執行；`test_execute_zh_to_en_translates_the_simplified_input` | engine 收到 `to_simplified` 的回傳值 | None |
| UT-05 | Design 3.4 #2 | Equivalence Partitioning | `zh→en` 執行；`test_execute_zh_to_en_returns_engine_output_without_script_conversion` | `target_text` 等於 engine 的輸出，`to_traditional` 沒有被呼叫 | None |
| UT-06 | Design 3.3、3.4 的 `source_text` 規則 | Equivalence Partitioning | 兩個方向參數化（`en-to-zh`、`zh-to-en`）；`test_execute_source_text_is_the_original_input` | `source_text` 等於傳入的原文，不是轉換後的中間結果 | None |
| UT-07 | ADR-0010 | Equivalence Partitioning | 兩個方向參數化；`test_execute_calls_translate_exactly_once` | engine 的 `translate` 被呼叫 1 次 | None |
| UT-08 | 引擎例外原樣往上拋 | Equivalence Partitioning | engine 拋出 `TranslationInputTooLongError`；`test_execute_engine_error_propagates_unchanged` | 同一個例外往上拋，沒有被包裝 | None |
| UT-09 | ADR-0003 | Equivalence Partitioning | 兩個語言不同；`test_translation_request_distinct_languages_accepts` | 建立成功，欄位等於輸入 | None |
| UT-10 | ADR-0003 | Equivalence Partitioning | 兩個語言相同，參數化（`en-en`、`zh-zh`）；`test_translation_request_source_equals_target_raises_validation_error` | 拋出 `ValidationError` | None |
| UT-11 | `text` 至少 1 個字元 | Boundary Value Analysis | `text` 為空字串；`test_translation_request_empty_text_raises_validation_error` | 拋出 `ValidationError` | None |
| UT-12 | `text` 至少 1 個字元 | Boundary Value Analysis | `text` 為 1 個字元；`test_translation_request_single_character_text_accepts` | 建立成功 | None |
| UT-13 | Design 3.6 HTTP 翻譯 | Equivalence Partitioning | stub use case 回傳固定結果，POST 合法請求；`test_translate_text_valid_request_returns_target_text` | status 200，JSON 含 `source_text`、`target_text`、`source_lang`、`target_lang`，值等於 stub 的結果 | 不驗證真實翻譯 |
| UT-14 | ADR-0003 | Equivalence Partitioning | POST 兩個語言相同的請求；`test_translate_text_source_equals_target_returns_422` | status 422，stub use case 未被呼叫 | None |
| UT-15 | 輸入過長 → HTTP 422 | Equivalence Partitioning | stub use case 拋出 `TranslationInputTooLongError`；`test_translate_text_input_too_long_returns_422` | status 422 | None |
| UT-16 | 未預期例外 → HTTP 500 | Equivalence Partitioning | stub use case 拋出 `RuntimeError("boom")`；`test_translate_text_unexpected_exception_returns_500` | status 500，`detail` 含 `boom` | 不驗證堆疊記錄的內容 |
| UT-17 | Design 3.2 #1 至 #2 | Equivalence Partitioning | 以 monkeypatch 把 `build_translate_text_use_case` 換成計數的 factory，進入 `TestClient` 後送出 2 個請求；`test_lifespan_builds_the_use_case_once_for_many_requests` | factory 被呼叫 1 次，兩個請求都得到 200 | None |
| UT-18 | Design 3.1 #3；Design 3.6 CLI 英譯中 | Equivalence Partitioning | stub use case 回傳固定結果；`test_main_valid_arguments_returns_zero_with_target_text_on_stdout` | exit code 0，stdout 為 `target_text` 加換行 | 不驗證真實翻譯 |
| UT-19 | ADR-0003 | Equivalence Partitioning | 兩個語言相同；`test_main_source_equals_target_returns_one_with_message_on_stderr` | exit code 1，stderr 含 `source_lang and target_lang must differ`，stub 未被呼叫 | None |
| UT-20 | 輸入過長 → CLI exit code 1 | Equivalence Partitioning | stub use case 拋出 `TranslationInputTooLongError`；`test_main_input_too_long_returns_nonzero_exit_code` | exit code 1，stderr 非空 | None |
| UT-21 | Design 2.2 `ScriptConverter` | Equivalence Partitioning | 繁體輸入；`test_to_simplified_converts_traditional_characters` | `步兵戰車` 轉為 `步兵战车` | None |
| UT-22 | Design 2.2 `ScriptConverter` | Equivalence Partitioning | 簡體輸入；`test_to_simplified_leaves_already_simplified_text_unchanged` | 輸出等於輸入 | None |
| UT-23 | Design 3.3 #2 | Equivalence Partitioning | 簡體輸入；`test_to_traditional_converts_simplified_characters` | `猫很快` 轉為 `貓很快` | None |
| UT-24 | Design 3.3 #2 | Equivalence Partitioning | 已是繁體的輸入；`test_to_traditional_leaves_already_traditional_text_unchanged` | 輸出等於輸入 | None |
| UT-25 | Design 2.2 `ScriptConverter` | Boundary Value Analysis | 空字串，兩個方法參數化（`to-simplified`、`to-traditional`）；`test_script_conversion_empty_string_returns_empty_string` | 回傳空字串 | None |
| UT-26 | ADR-0004 | Equivalence Partitioning | 以 monkeypatch 清除兩個環境變數後重新載入 `_offline_env`；`test_offline_env_import_sets_both_variables_to_one_by_default` | `HF_HUB_OFFLINE` 與 `TRANSFORMERS_OFFLINE` 皆為 `1` | 不驗證 `transformers` 匯入時真的讀到該值 |
| UT-27 | ADR-0004 | Equivalence Partitioning | 以 monkeypatch 先設 `HF_HUB_OFFLINE` 為 `0` 後重新載入；`test_offline_env_import_keeps_existing_values` | `HF_HUB_OFFLINE` 仍為 `0` | None |
| UT-28 | ADR-0010 | Equivalence Partitioning | 以 `inspect.signature` 檢查建構子；`test_use_case_constructor_parameters_are_engine_then_converter` | 參數名稱依序為 `translation_engine`、`script_converter` | 只驗證建構子，不驗證 `bootstrap.py` 沒有讀詞彙表檔（見 ET-01） |
| UT-29 | 缺少權重 → CLI exit code 1 | Equivalence Partitioning | 以 monkeypatch 把 `build_translate_text_use_case` 換成拋出 `RuntimeError("Model binaries missing for zh")` 的函式，執行 `main`；`test_main_startup_runtime_error_returns_one_with_message_on_stderr` | exit code 1，stderr 含 `Model binaries missing` | 不驗證真實缺少權重的情形（見 IT-06） |
| UT-30 | 缺少權重 → API 伺服器啟動失敗 | Equivalence Partitioning | 同樣的 monkeypatch，進入 `TestClient(app)` 的 lifespan；`test_lifespan_startup_failure_propagates_runtime_error` | 拋出 `RuntimeError`，訊息含 `Model binaries missing` | None |

### 3.2 Integration

| Test ID | Claim | Technique | Procedure | Expected Result | Not Covered |
|---|---|---|---|---|---|
| IT-01 | Design 2.1 `CTranslate2Adapter` | Equivalence Partitioning | 以真實權重翻譯一句短英文；`test_translate_short_text_returns_nonempty_target_text` | `target_text` 為非空字串，`source_lang`、`target_lang` 等於輸入 | 不驗證翻譯品質 |
| IT-02 | 512-token 上限 | Boundary Value Analysis | 輸入斷詞後恰好 512 個 token；`test_translate_text_at_512_tokens_succeeds` | 回傳 `TranslationResult`，不拋例外 | None |
| IT-03 | 512-token 上限 | Boundary Value Analysis | 輸入斷詞後 513 個 token；`test_translate_text_over_512_tokens_raises_translation_input_too_long_error` | 拋出 `TranslationInputTooLongError` | None |
| IT-04 | Design 3.3 #1 至 #2 | Equivalence Partitioning | 真實 `CTranslate2Adapter` 與 `OpenCcAdapter` 組成 use case，`en→zh` 翻譯 `Their commercial space sector features partially state-owned enterprises.`；`test_execute_real_adapters_en_to_zh_returns_traditional_text` | `target_text` 非空，且 `to_traditional(target_text)` 等於 `target_text` | 不驗證含特定譯名 |
| IT-05 | Design 3.4 #1 至 #2 | Equivalence Partitioning | 同上，`zh→en` 翻譯繁體輸入 `其商業太空部門，包括遙感探測、發射與通信服務，具有部分國有的企業`；`test_execute_real_adapters_zh_to_en_traditional_input_returns_english` | `target_text` 含至少一個 ASCII 字母，不含 CJK 字元 | 不驗證含特定譯名 |
| IT-06 | Design 2.1 資源所有權；缺少任一方向的權重時建構失敗 | Equivalence Partitioning | 在 `tmp_path` 只複製 `models/` 的 `opus-mt-en-zh` 與 `opus-mt-en-zh-ct2`，缺少 `zh→en` 的兩個目錄；建構 `CTranslate2Adapter(models_dir=<tmp_path>)`；`test_adapter_missing_one_direction_raises_runtime_error` | 拋出 `RuntimeError`，訊息含 `Model binaries missing` | 不驗證 CLI 與 API 對這個錯誤的處理（見 UT-29、UT-30） |

### 3.3 Environment

| Test ID | Claim | Technique | Procedure | Expected Result | Not Covered |
|---|---|---|---|---|---|
| ET-01 | Design 3.1；Design 3.6 CLI 英譯中；ADR-0010 | State Transition | 在沒有 `data/glossary_*.json` 的專案根目錄，以真實權重呼叫 `main(["translate", "-s", "en", "-t", "zh", "Their commercial space sector features partially state-owned enterprises."])`；`test_cli_real_run_en_to_zh_prints_one_traditional_line` | exit code 0，stdout 一行非空文字，`to_traditional` 不改變它 | 不驗證含特定譯名 |
| ET-02 | Design 3.1；Design 3.6 CLI 中譯英 | State Transition | 同上，`zh→en`；`test_cli_real_run_zh_to_en_prints_one_english_line` | exit code 0，stdout 一行非空英文 | 不驗證含特定譯名 |
| ET-03 | Design 3.6 HTTP 翻譯；Design 3.2 | State Transition | 以真實權重進入 `TestClient(app)` 的 lifespan，POST 一個合法請求；`test_http_real_run_returns_translation_result_fields` | status 200，JSON 的 `source_text` 等於輸入，`target_text` 非空 | 不驗證 uvicorn 的實際監聽埠 |
| ET-04 | Design 3.5 | State Transition | 人工：API 未啟動時開啟 Streamlit 頁面並按 Translate；沒有測試函式 | 畫面顯示連線錯誤訊息，不顯示 `target_text` | 非連線失敗的三種分支（非 JSON、`ok` 為 false、缺 `target_text`）以人工操作難以重現；版面與導覽未驗證 |
| ET-05 | Design 3.6 UI 翻譯 | State Transition | 人工：啟動 API 與 Streamlit，輸入英文並按 Translate；沒有測試函式 | 畫面顯示 `target_text` | 版面、下拉選單互動、生命週期未驗證 |
