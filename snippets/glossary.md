<!-- GENERATED from data/glossary.yml by scripts/generate_snippets.py -- do not edit. -->

*[MTP]: 多詞元預測 — 小型草稿模型一次提出多個詞元，主模型一次驗證通過
*[QAT]: 量化感知訓練 — 訓練時模擬量化，讓模型學會補償量化造成的影響
*[PTQ]: 訓練後量化 — 壓縮已訓練完成的模型，在相同位元寬度下品質損失比 QAT 大
*[GGUF]: llama.cpp 及其生態系用於量化權重的檔案格式
*[LoRA]: 低秩適應 — 只微調新增的一小組權重，而非整個模型
*[QLoRA]: 在量化基礎模型上套用 LoRA，讓微調能在單張消費級 GPU 上執行
*[SFT]: 監督式微調 — 使用提示與回應配對進行訓練
*[DPO]: 直接偏好最佳化 — 從偏好與拒絕的回應配對中進行對齊
*[MoE]: 混合專家 — 對於任何詞元，只有部分模型參數會被啟用
*[PLE]: 逐層嵌入 — 嵌入表存放在快閃儲存體而非記憶體中
*[KV cache]: 已處理詞元所儲存的注意力鍵與值；會隨上下文長度增長
*[IT]: 指令微調 — 經訓練能遵循指令，無需進一步訓練即可給出有用的回應
*[PT]: 預訓練 — 指令微調之前的基礎模型
*[VLM]: 視覺語言模型 — 可同時接收圖像與文字的模型
*[RAG]: 檢索增強生成 — 檢索相關文件並放入提示中
*[TPU]: 張量處理單元 — Google 用於機器學習工作負載的加速器
*[PEFT]: Parameter-Efficient Fine-Tuning — tuning a small subset of parameters rather than updating all model weights
*[FT]: Fine-Tuned — specialized for a specific benchmark, dataset, or domain task
*[ASR]: Automatic Speech Recognition — transcribing spoken audio into text
*[VQA]: Visual Question Answering — answering questions about visual content in an image
*[OCR]: Optical Character Recognition — extracting written or printed text directly from images
*[RLHF]: Reinforcement Learning from Human Feedback — aligning model behavior with human preferences
