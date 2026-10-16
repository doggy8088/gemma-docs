---
title: Gemini API
description: 透過 Gemini API 呼叫託管的 Gemma 4 模型，而無需自行執行與服務執行個體。
models: [gemma-4]
frameworks: [gemini-api]
source_url: "https://ai.google.dev/gemma/docs/core/gemma_on_gemini_api.md.txt"
---

# 使用 Gemini API 執行 Gemma

Gemini API 提供託管的 Gemma 存取，作為可在應用程式開發或原型設計中使用的程式化 API。
此 API 是自行架設 Gemma 本機執行個體與處理生成式 AI 任務的 Web 服務的便利替代方案。

## 支援的模型

Gemini API 託管 Gemma 4 中最大的兩個模型：

| 模型 | 識別碼 |
|---|---|
| Gemma 4 31B | `gemma-4-31b-it` |
| Gemma 4 26B A4B | `gemma-4-26b-a4b-it` |

較小的尺寸沒有託管——請使用 [Ollama](ollama.md)、
[llama.cpp](llamacpp.md) 或 [LiteRT-LM](litert.md) 自行執行。本頁的選取器提供 API 所服務的兩種
尺寸，下方的每個範例都遵循它。

=== "Python"

    ```python { .model-swap }
    from google import genai

    client = genai.Client()

    response = client.models.generate_content(
        model="gemma-4-26b-a4b-it",
        contents="Roses are red...",
    )

    print(response.text)
    ```

=== "JavaScript"

    ```javascript { .model-swap }
    import { GoogleGenAI } from "@google/genai";

    const ai = new GoogleGenAI();

    const response = await ai.models.generateContent({
      model: "gemma-4-26b-a4b-it",
      contents: "Roses are red...",
    });
    console.log(response.text);
    ```

=== "REST"

    ```bash { .model-swap }
    curl "https://generativelanguage.googleapis.com/v1beta/models/gemma-4-26b-a4b-it:generateContent" \
    -H 'Content-Type: application/json' \
    -X POST \
    -d '{
      "contents": [{
        "parts":[{"text": "Roses are red..."}]
        }]
       }'
    ```

[取得 API 金鑰](https://aistudio.google.com/apikey)

!!! Important
    您必須取得 API 金鑰才能使用 Gemini API，可從 Google [AI Studio](https://aistudio.google.com/apikey) 應用程式取得。

您可以在許多平台上存取 Gemini API，例如行動裝置、網頁與雲端
服務，並支援多種程式語言。關於 Gemini API SDK 套件的更多資訊，請參閱 Gemini API
[SDK 下載](https://ai.google.dev/gemini-api/docs/downloads)
頁面。關於 Gemini API 的一般介紹，請參閱
[Gemini API 入門指南](https://ai.google.dev/gemini-api/docs/get-started)。

## 思考

Gemma 4 運用內部的「思考流程」來最佳化其多步
推理，在演算法程式設計與高等數學證明等邏輯要求嚴苛的領域
提供卓越的效能。

雖然 Gemma 4 僅支援開啟或關閉此功能，但您可以透過將思考等級設為 `"high"`（啟用）或
`"minimal"`（停用）來使用 API 控制它。

以下範例示範如何啟動思考流程：

=== "Python"

    ```python { .model-swap }
    from google import genai
    from google.genai import types

    client = genai.Client()

    response = client.models.generate_content(
        model="gemma-4-26b-a4b-it",
        contents="What is the water formula?",
        config=types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_level="high")
        ),
    )

    print(response.text)
    ```

=== "JavaScript"

    ```javascript { .model-swap }
    import { GoogleGenAI, ThinkingLevel } from "@google/genai";

    const ai = new GoogleGenAI();

    const response = await ai.models.generateContent({
      model: "gemma-4-26b-a4b-it",
      contents: "What is the water formula?",
      config: {
        thinkingConfig: {
          thinkingLevel: ThinkingLevel.HIGH,
        },
      },
    });
    console.log(response.text);
    ```

=== "REST"

    ```bash { .model-swap }
    curl "https://generativelanguage.googleapis.com/v1beta/models/gemma-4-26b-a4b-it:generateContent" \
    -H 'Content-Type: application/json' \
    -X POST \
    -d '{
      "contents": [{
        "parts":[{"text": "What is the water formula?"}]
        }],
        "generationConfig": {
          "thinkingConfig": {
                "thinkingLevel": "high"
          }
        }
       }'
    ```

深入了解思考：

- [Gemini API 思考](https://ai.google.dev/gemini-api/docs/thinking)（一般介紹）
- [Gemma 思考](../get-started/capabilities/thinking.md)（Gemma 專屬功能）

## 圖像理解

Gemma 4 模型可以處理圖像，為開發者開啟了許多前沿的使用案例，
這些案例過去需要特定領域的模型。

以下範例示範如何在 Gemini API 中使用 Gemma 圖像輸入：

=== "Python"

    ```python { .model-swap }
    from google import genai

    client = genai.Client()

    my_file = client.files.upload(file="path/to/sample.jpg")

    response = client.models.generate_content(
        model="gemma-4-26b-a4b-it",
        contents=[my_file, "Caption this image."],
    )

    print(response.text)
    ```

=== "JavaScript"

    ```javascript { .model-swap }
    import {
      GoogleGenAI,
      createUserContent,
      createPartFromUri,
    } from "@google/genai";

    const ai = new GoogleGenAI();

    const myfile = await ai.files.upload({
      file: "path/to/sample.jpg",
      config: { mimeType: "image/jpeg" },
    });

    const response = await ai.models.generateContent({
      model: "gemma-4-26b-a4b-it",
      contents: createUserContent([
        createPartFromUri(myfile.uri, myfile.mimeType),
        "Caption this image.",
      ]),
    });
    console.log(response.text);
    ```

=== "REST"

    ```bash { .model-swap }
    IMAGE_PATH="cats-and-dogs.jpg"
    MIME_TYPE=$(file -b --mime-type "${IMAGE_PATH}")
    NUM_BYTES=$(wc -c < "${IMAGE_PATH}")
    DISPLAY_NAME=IMAGE

    tmp_header_file=upload-header.tmp

    # Initial resumable request defining metadata.
    curl "https://generativelanguage.googleapis.com/upload/v1beta/files" \
      -D upload-header.tmp \
      -H "X-Goog-Upload-Protocol: resumable" \
      -H "X-Goog-Upload-Command: start" \
      -H "X-Goog-Upload-Header-Content-Length: ${NUM_BYTES}" \
      -H "X-Goog-Upload-Header-Content-Type: ${MIME_TYPE}" \
      -H "Content-Type: application/json" \
      -d "{'file': {'display_name': '${DISPLAY_NAME}'}}" 2> /dev/null

    upload_url=$(grep -i "x-goog-upload-url: " "${tmp_header_file}" | cut -d" " -f2 | tr -d "\r")
    rm "${tmp_header_file}"

    # Upload the actual bytes.
    curl "${upload_url}" \
      -H "Content-Length: ${NUM_BYTES}" \
      -H "X-Goog-Upload-Offset: 0" \
      -H "X-Goog-Upload-Command: upload, finalize" \
      --data-binary "@${IMAGE_PATH}" 2> /dev/null > file_info.json

    file_uri=$(jq -r ".file.uri" file_info.json)
    echo file_uri=$file_uri

    # Now generate content using that file
    curl "https://generativelanguage.googleapis.com/v1beta/models/gemma-4-26b-a4b-it:generateContent" \
        -H 'Content-Type: application/json' \
        -X POST \
        -d '{
          "contents": [{
            "parts":[
              {"file_data":{"mime_type": "'"${MIME_TYPE}"'", "file_uri": "'"${file_uri}"'"}},
              {"text": "Caption this image."}]
            }]
          }' 2> /dev/null > response.json

    cat response.json
    echo

    jq -r ".candidates[].content.parts[].text" response.json
    ```

深入了解圖像理解：

- [Gemini API 圖像理解](https://ai.google.dev/gemini-api/docs/image-understanding)（一般介紹）
- [Gemma 圖像理解](../get-started/capabilities/image.md)（Gemma 專屬功能）

## 系統指示

您可以傳入系統指示來設定模型的行為：

=== "Python"

    ```python { .model-swap }
    from google import genai
    from google.genai import types

    client = genai.Client()

    response = client.models.generate_content(
        model="gemma-4-26b-a4b-it",
        config=types.GenerateContentConfig(
            system_instruction="You are a wise Kyoto tea master. Speak calmly and poetically, using nature metaphors. Keep answers under 3 sentences."
        ),
        contents="What is the purpose of the tea ceremony?"
    )
    print(response.text)
    ```

=== "JavaScript"

    ```javascript { .model-swap }
    import { GoogleGenAI } from "@google/genai";

    const ai = new GoogleGenAI();

    const response = await ai.models.generateContent({
      model: "gemma-4-26b-a4b-it",
      contents: "What is the purpose of the tea ceremony?",
      config: {
        systemInstruction: "You are a wise Kyoto tea master. Speak calmly and poetically, using nature metaphors. Keep answers under 3 sentences."
      }
    });
    console.log(response.text);
    ```

=== "REST"

    ```bash { .model-swap }
    curl "https://generativelanguage.googleapis.com/v1beta/models/gemma-4-26b-a4b-it:generateContent" \
    -H 'Content-Type: application/json' \
    -X POST \
    -d '{
      "contents": [{
        "parts":[{"text": "What is the purpose of the tea ceremony?"}]
      }],
      "systemInstruction": {
        "parts": [{"text": "You are a wise Kyoto tea master. Speak calmly and poetically, using nature metaphors. Keep answers under 3 sentences."}]
      }
    }'
    ```

## 多輪對話

SDK 提供了自動追蹤對話歷史的聊天介面：

=== "Python"

    ```python { .model-swap }
    from google import genai

    client = genai.Client()
    chat = client.chats.create(model="gemma-4-26b-a4b-it")

    response = chat.send_message("What are the three most famous castles in Japan?")
    print(response.text)

    response = chat.send_message("Which one should I visit in spring for cherry blossoms?")
    print(response.text)
    ```

=== "JavaScript"

    ```javascript { .model-swap }
    import { GoogleGenAI } from "@google/genai";

    const ai = new GoogleGenAI();
    const chat = ai.chats.create({ model: "gemma-4-26b-a4b-it" });

    let response = await chat.sendMessage({ message: "What are the three most famous castles in Japan?" });
    console.log(response.text);

    response = await chat.sendMessage({ message: "Which one should I visit in spring for cherry blossoms?" });
    console.log(response.text);
    ```

=== "REST"

    ```bash { .model-swap }
    curl "https://generativelanguage.googleapis.com/v1beta/models/gemma-4-26b-a4b-it:generateContent" \
    -H 'Content-Type: application/json' \
    -X POST \
    -d '{
      "contents": [
        {
          "role": "user",
          "parts": [{ "text": "What are the three most famous castles in Japan?" }]
        },
        {
          "role": "model",
          "parts": [{ "text": "Himeji Castle, Matsumoto Castle, and Kumamoto Castle are often considered the top three." }]
        },
        {
          "role": "user",
          "parts": [{ "text": "Which one should I visit in spring for cherry blossoms?" }]
        }
      ]
    }'
    ```

## 函數呼叫

將工具定義為函數宣告。模型會決定何時呼叫它們：

=== "Python"

    ```python { .model-swap }
    from google import genai
    from google.genai import types

    # Define the function declaration
    get_weather = {
        "name": "get_weather",
        "description": "Get current weather for a given location.",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City and state, e.g. 'San Francisco, CA'",
                },
            },
            "required": ["location"],
        },
    }

    client = genai.Client()
    tools = types.Tool(function_declarations=[get_weather])
    config = types.GenerateContentConfig(tools=[tools])

    response = client.models.generate_content(
        model="gemma-4-26b-a4b-it",
        contents="Should I bring an umbrella to Kyoto today?",
        config=config,
    )

    # The model returns a function call instead of text
    if response.function_calls:
        for fc in response.function_calls:
            print(f"Function to call: {fc.name}")
            print(f"ID: {fc.id}")
            print(f"Arguments: {fc.args}")
    else:
        print("No function call found in the response.")
        print(response.text)
    ```

=== "JavaScript"

    ```javascript { .model-swap }
    import { GoogleGenAI } from "@google/genai";

    const ai = new GoogleGenAI();

    const get_weather = {
        name: "get_weather",
        description: "Get current weather for a given location.",
        parameters: {
            type: "object",
            properties: {
                location: {
                    type: "string",
                    description: "City and state, e.g. 'San Francisco, CA'",
                },
            },
            required: ["location"],
        },
    };

    const response = await ai.models.generateContent({
      model: "gemma-4-26b-a4b-it",
      contents: "Should I bring an umbrella to Kyoto today?",
      config: {
        tools: [{ functionDeclarations: [get_weather] }]
      }
    });

    if (response.functionCalls) {
        for (const fc of response.functionCalls) {
            console.log(`Function to call: ${fc.name}`);
            console.log(`Arguments: ${JSON.stringify(fc.args)}`);
        }
    } else {
        console.log("No function call found in the response.");
        console.log(response.text);
    }
    ```

=== "REST"

    ```bash { .model-swap }
    curl "https://generativelanguage.googleapis.com/v1beta/models/gemma-4-26b-a4b-it:generateContent" \
    -H 'Content-Type: application/json' \
    -X POST \
    -d '{
      "contents": [{
        "parts":[{"text": "Should I bring an umbrella to Kyoto today?"}]
      }],
      "tools": [{
        "functionDeclarations": [{
          "name": "get_weather",
          "description": "Get current weather for a given location.",
          "parameters": {
            "type": "OBJECT",
            "properties": {
              "location": {
                "type": "STRING",
                "description": "City and state, e.g. 'San Francisco, CA'"
              }
            },
            "required": ["location"]
          }
        }]
      }]
    }'
    ```

## Google 搜尋

以 Google 搜尋的即時網路資料為 Gemma 4 回應接地：

=== "Python"

    ```python { .model-swap }
    from google import genai
    from google.genai import types

    client = genai.Client()

    response = client.models.generate_content(
        model="gemma-4-26b-a4b-it",
        contents="What are the dates for cherry blossom season in Tokyo this year?",
        config=types.GenerateContentConfig(
            tools=[{"google_search":{}}]
        ),
    )

    print(response.text)

    # Access grounding metadata for citations
    for chunk in response.candidates[0].grounding_metadata.grounding_chunks:
        print(f"Source: {chunk.web.title} --- {chunk.web.uri}")
    ```

=== "JavaScript"

    ```javascript { .model-swap }
    import { GoogleGenAI } from "@google/genai";

    const ai = new GoogleGenAI();

    const response = await ai.models.generateContent({
      model: "gemma-4-26b-a4b-it",
      contents: "What are the dates for cherry blossom season in Tokyo this year?",
      config: {
        tools: [{ googleSearch: {} }]
      }
    });

    console.log(response.text);

    if (response.candidates?.[0]?.groundingMetadata?.groundingChunks) {
        for (const chunk of response.candidates[0].groundingMetadata.groundingChunks) {
            if (chunk.web) {
                console.log(`Source: ${chunk.web.title} --- ${chunk.web.uri}`);
            }
        }
    }
    ```

=== "REST"

    ```bash { .model-swap }
    curl "https://generativelanguage.googleapis.com/v1beta/models/gemma-4-26b-a4b-it:generateContent" \
    -H 'Content-Type: application/json' \
    -X POST \
    -d '{
      "contents": [{
        "parts":[{"text": "What are the dates for cherry blossom season in Tokyo this year?"}]
      }],
      "tools": [{"googleSearch": {}}]
    }'
    ```
