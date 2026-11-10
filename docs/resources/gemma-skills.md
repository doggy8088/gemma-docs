---
title: Gemma Skills
description: 教導程式設計代理如何使用 Gemma 打造應用程式，以及如何微調 Gemma 的代理技能。
models: [gemma-4]
---

# Gemma Skills

[Gemma Skills][gemma-skills-repo] 是可安裝的技能，讓程式設計代理具備使用 Gemma 的實戰知識——最新的模型名稱、該使用的 API，以及任務實際需要的步驟。安裝之後，代理就不再需要憑空猜測。

<div class="card-grid">
<a class="card" href="https://github.com/google-gemma/gemma-skills/tree/main/skills/gemma-dev/SKILL.md" target="_blank" rel="noopener">
  <div class="card-title">gemma-dev <span class="card-ext">↗</span></div>
  <p class="card-desc">使用 Gemma 打造應用程式，以及關於模型的一般問題。</p>
</a>
<a class="card" href="https://github.com/google-gemma/gemma-skills/tree/main/skills/gemma-trainer/SKILL.md" target="_blank" rel="noopener">
  <div class="card-title">gemma-trainer <span class="card-ext">↗</span></div>
  <p class="card-desc">在本機硬體上訓練與調整 Gemma——SFT、DPO、RLHF 與獎勵建模。</p>
</a>
</div>

## 安裝

=== "skills CLI"

    ```bash
    # Interactively browse and install skills.
    npx skills add google-gemma/gemma-skills --list

    # Install a specific skill (e.g., gemma-dev).
    npx skills add google-gemma/gemma-skills --skill gemma-dev
    ```

=== "Antigravity CLI"

    ```bash
    # Install it directly
    agy plugin install https://github.com/google-gemma/gemma-skills
    ```

!!! note
    此存放庫採用 Apache-2.0 授權，並非 Google 官方支援的產品。
