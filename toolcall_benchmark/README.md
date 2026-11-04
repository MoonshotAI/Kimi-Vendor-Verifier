# toolcall_benchmark JSONL Data Description

This directory contains sample data for tool-call capability evaluation in **JSONL (JSON Lines)** format:

- Each line is an independent JSON object representing a complete model conversation sample.
- Lines share the same schema and can be parsed line by line with `json.loads(line)`.
- Only the thinking benchmark is provided: newer models have no non-thinking requirement, so there is no non-thinking tool-call dataset.

> Note: This dataset has been fully de-identified and is safe to share. Credentials and
> tokens are replaced with typed placeholders (e.g. `[REDACTED_SECRET]`); phone numbers,
> emails, addresses, IPs, platform IDs and home-path usernames are replaced with
> consistent synthetic or masked values (e.g. `188****2656`, `uNNNN@example.com`, 
> [TELEGRAM_ID], `user9727`). Person names are de-identified by replacement with novel 
> character names; see below.
>
> **Security properties of identifier replacement.** Platform IDs (Feishu, Discord
> snowflakes, Telegram numeric IDs, etc.) carry their full identifying entropy in
> essentially every character: masking only a suffix or prefix leaves a unique,
> stable pseudonymous identifier that remains linkable across datasets and
> searchable in public logs, i.e. the data subject is still re-identifiable.
> These identifiers are therefore replaced with HMAC-derived synthetics: without
> the (local-only, unpublished) key, no holder can recover or link back to any
> original value, so the published dataset is unlinkable to the source identities
> by construction. The synthetic values keep the original format only to satisfy
> downstream format validation; they carry no information about the data subject
> themselves.

## Top-level Fields per Line

| Field | Type | Description |
|------|------|------|
| `id` | `str` | Unique sample identifier (UUID). |
| `created_at` | `str` | Sample creation time with timezone, e.g. `2026-04-26 11:59:21+08:00`. |
| `prompt_tokens` | `int` | Number of input prompt tokens. |
| `completion_tokens` | `int` | Number of model-generated output tokens. |
| `cached_tokens` | `int` | Number of cached tokens hit. |
| `total_tokens` | `int` | Total token count (approximately `prompt_tokens + completion_tokens`). |
| `msg_count` | `int` | Total number of messages in the conversation. |
| `_repeat_run` | `dict` | Repeat/loop tool-call analysis result; see below. |
| `tools` | `list` | List of available tools in this conversation; each element is an OpenAI-style function tool definition. |
| `thinking` | `dict` | Thinking-mode configuration, containing only `type`: `enabled` or `disabled`. |
| `messages` | `list` | Full conversation history; each element is a message object; see below. |

## `_repeat_run` Field

Annotates whether the sample exhibits an abnormal pattern of repeatedly calling the same tool. This data originates from online error reports, so this field records the error locations and contents; kimi_toolcall consumes the data according to the indices documented below.

| Field | Type | Description |
|------|------|------|
| `tool_name` | `str` | Name of the tool identified as being called repeatedly. |
| `run_first_idx` | `int` | Index in `messages` of the first normal invocation of the tool. |
| `repeat_start_idx` | `int` | Start index of the repeated invocation segment. |
| `run_last_idx` | `int` | End index of the repeated invocation segment. |
| `run_length` | `int` | Number of messages in the repeated invocation segment. |
| `loop_phrase_hits` | `int` | Count of detected loop phrases. |
| `error_phrase_hits` | `int` | Count of detected error phrases. |
| `tool_result_error_count` | `int` | Number of times the tool returned an error. |
| `tool_result_total` | `int` | Total number of invocations of the tool. |
| `tool_result_error_ratio` | `float` | Ratio of tool results that were errors. |
| `sample_error_excerpts` | `list` | Exemplar error excerpts; each item contains `msg_idx` and `excerpt`. |

## `tools` Field

A standard OpenAI function tool array. Each element has the following format:

```json
{
  "type": "function",
  "function": {
    "name": "read",
    "description": "...",
    "parameters": {
      "type": "object",
      "properties": { ... },
      "required": ["path"]
    }
  }
}
```

## `messages` Field

Array of conversation messages in chronological order. Common roles and fields:

| `role` | Common Fields | Description |
|--------|---------------|-------------|
| `user` | `role`, `content` | User input. |
| `assistant` | `role`, `content`, `tool_calls`, `reasoning_content` | Model response. `tool_calls` is the list of tool calls initiated by the model; `reasoning_content` is the reasoning content produced by thinking-capable models (usually absent in non-thinking samples). |
| `tool` | `role`, `content`, `tool_call_id` | Tool execution result; `tool_call_id` corresponds to `assistant.tool_calls[].id`. |

Each element in the `assistant.tool_calls` array contains:

| Field | Description |
|------|-------------|
| `id` | Tool-call ID, used by the `tool` message to return the result. |
| `type` | Always `function`. |
| `function` | Contains `name` (tool name) and `arguments` (parameters as a JSON string). |

## Example Usage

```python
import json

with open("toolcall_benchmark/toolcall_thinking_samples.stable2.jsonl") as f:
    for line in f:
        sample = json.loads(line)
        print(sample["id"], sample["thinking"])
        for msg in sample["messages"][:5]:
            print(" ", msg["role"], msg.keys())
```

## Person Name De-identification (Novel Character Naming)
All person names in `toolcall_thinking_samples.stable2.jsonl` (70 samples) have been replaced with fictional character names — Chinese names with characters from *Water Margin* (水浒传), English names with characters from *War and Peace* (战争与和平). This covers real names(includes team templates, blogger handles, textbook credits), without public figures in news content. 
Rules:
- One person = one fictional identity; all forms of the same person (full name / given / family / alias / case variants) map to one character. For public figures mentioned in both scripts within one sample, all forms unify to one *War and Peace* name; in Chinese-only samples they take a *Water Margin* name.
- The same fictional name may designate different real persons in different samples (samples are independent).
- Sample-scoped replacement; Latin names use word-boundary matching; masked forms (`X*`) exclude Markdown bold adjacency.
Exceptions (kept as-is):
- Synthetic IDs: `user8402`, `user3617`, `user4547`, `user0041`, `user9727`.


Only the novel character names used by each sample are listed here:

| Sample | Novel character names used (incl. places) |
|---|---|
| 07e927dc | 宋江、卢俊义、吴用、公孙胜、关胜、林冲、秦明、呼延灼、Pierre Bezukhov |
| 0fbec37d | Speranski、Langeron、Alexander Langeron、Dokhturov、Berthier、Bessieres、Oudinot、Rapp、Savary、Friant、Pierre、Nikolai Rostov、Platon Karataev、Balashev、Murat、Bagration |
| 103404ca | 郁保四、鲁智深、Lu Zhishen、ZhiShen Lu、Sergei Wolzogen、A. Speranski、Vasily Balashev、Z. Zherkov、M. Willarski、Ilya Ermolov、Daniel Oudinot、Heming Bilibin、Haibo Tushin、Nikolai Rostov、Andrei Bolkonsky、Pierre Bezukhov、Boris Drubetskoy、Anatole Kuragin、Fedor Dolokhov、Platon Karataev、Hippolyte Kuragin、Kuzma Timokhin、zhishen-lu、davout、berthier |
| 15c998ea | 白胜、段景住、高俅、蔡京 |
| 24818454 | Dmitri Rostov、Kirill Rostov、Anton Karagin、Roman Karagin、Semyon Kuragin、Taras Dolokhov、Prokhor Denisov、Efim Tushin、Kuzma Timokhin、Gavrilo Berg、Pavel Bilibin、Stepan Nesvitsky、YURI ZHERKOV、Lev Raevsky、Adrian Pfuel、Matvey Weyrother、Osip Langeron、Innokenty Dokhturov、Foma Ermolov、Hariton Miloradovich、Akim Murat、Viktor Davout、Platon Ney、Zakhar Berthier、Naum Caulaincourt、Lazar Bessieres、Orest Oudinot、Karp Rapp、Fadei Savary、Uvar Friant、Vlas Mortemart、Yefim Willarski、Anatole、Andrei Bolkonsky、Pelageya、Pierre Bezukhov |
| 2560e391 | 索超、戴宗、刘唐、李逵、史进、穆弘、雷横、李俊、阮小二、张横、阮小五、张顺、阮小七、杨雄、石秀、解珍、解宝、燕青、朱武、黄信、孙立、宣赞、郝思文、韩滔、彭玘、单廷珪、魏定国、萧让、裴宣、欧鹏、邓飞、燕顺、杨林、凌振、蒋敬、吕方、郭盛、安道全、皇甫端、王英、扈三娘、鲍旭、樊瑞、孔明、孔亮、项充、李衮、金大坚、朱贵、朱富、蔡福、蔡庆、李立、李云、焦挺、石勇、孙新、张青、晁盖、林冲、武松、鲁智深、时迁、Berthier、Bessieres、Mortier、高俅 |
| 33328b30 | Karataev |
| 35f5ee72 | 花荣 |
| 36958f65 | Julie |
| 385a2538 | Andrei Bolkonsky、Andrei |
| 39b732e9 | Platov |
| 400383aa | 秦明、Anna Pavlovna |
| 42f7bd41 | 花荣、马麟、扈三娘、张清、顾大嫂、安道全、李俊 |
| 4af405be | 柴进、高俅 |
| 5c549feb | 柴进、高俅 |
| 5e5db8b1 | 李应 |
| 6e95bf06 | 朱仝、鲁智深、武松、董平、张清、杨志、王定六 |
| 6f9d5dd3 | 花荣、马麟、扈三娘、张清、顾大嫂、安道全、李俊 |
| 76fc47c5 | Dolokhov |
| 83a8aa02 | 徐宁、白胜、蔡京、Rostopchin、Lavrushka、Alpatych、Ilyin |
| 904336b5 | 吴用、朱武、柴进、宣赞、扈三娘、郭盛、林冲、秦明、呼延灼、关胜、徐宁、索超、戴宗、卢俊义、Wu Yong |
| 922c7d05 | 蔡京、李逵、燕青、武松 |
| 97f46ffc | 花荣、童威、童猛、孟康、侯健、陈达、杨春、郑天寿、陶宗旺、宋清、乐和、龚旺、丁得孙、穆春、曹正、宋万、杜迁、江州、沧州、Denisov、Hippolyte、安道全、索超、欧鹏、邓飞、顾大嫂、孙二娘、李师师、武松、林冲行记、鲁智深周末去哪儿、扈三娘、石秀、二龙山故事多、燕青周边游 |
| 98b2ac59 | 花荣 |
| a58a42ff | 薛永、施恩、朱仝、杨志、史进、邹润、Ilya Rostov、Compans、Gudin、段景住、库图佐夫、索尼娅、鲍里斯、多洛霍夫、宣赞 |
| a61d18d1 | 时迁、杨林、史文恭、王英、Bezukhov、扈三娘、孙二娘、李师师、顾大嫂 |
| a6d1edb7 | 卢俊义 |
| a741e205 | 花荣、Bolkonsky、Rostov |
| b5029d9e | 花荣、李忠 |
| b5e03ba7 | 周通、汤隆 |
| c221feb3 | 花荣、杜兴、邹渊、邹润、宋江、扈三娘、关胜、鲁智深、孙二娘、秦明、白秀英、吴用、金翠莲、顾大嫂、戴宗、琼英、李师师、燕青、李俊、Natasha Rostova、Natasha、Rostova、natasha、Marya Bolkonskaya、Marya、Bolkonskaya、玛丽亚、Helene Kuragina、Boris Drubetskoy、Pierre Bezukhov、Murat |
| d6f06ba7 | Annette Scherer、Vera Rostova、Speranski、Langeron、Dokhturov、Ermolov、Ney、Uvarov、Davout、Tormasov、Bourienne、Tushin、Dunyasha、Chichagov、Junot、Pierre Bezukhov、Andrei Bolkonsky、Marya Bolkonskaya、Boris Drubetskoy、Nikolai Rostov、Fedor Dolokhov、Karp Rapp、Pyotr Bagration、Kutuzov、Helene Kuragina、Yuri Zherkov、Vasily Balashev、Balashev、Sergei Wolzogen、Adrian Pfuel、Pfuel、Matvey Weyrother、Weyrother、Stepan Nesvitsky、Nesvitsky、Murat、Julie Karagina、Liza、Platon Karataev、Lev Raevsky、Berthier、Bessieres、Platov、Hippolyte、Kuzma Timokhin、Sonya、Oudinot、Catiche、Mavra、Anatole Dokhturov、Anatole、murat、bessieres、berthier、helene-kuragina、sonyas、liza、oudinot |
| db4da839 | Kutuzov、Liza、Ivan Timokhin、Louis Berthier、Pierre Bezukhov、dolokhov |
| ea89e4d7 | Anatole Kuragin、Anatole、Murat、Platon |
| ea904f1d | 段景住、高俅、穆弘、Kutuzov、Balashev、巴格拉季昂、海伦、多洛霍夫、拉耶夫斯基、罗斯托夫、皮埃尔、Pierre Bezukhov、蔡京 |
| f8caa478 | 武松 |
| eba40c49 | Pyotr Bagration、Adolf Berg、Alexander Bilibin、Fedor Dolokhov、Sonya Rostova、Speranski、Murat、Ney、Massena、Lannes、Duroc、Anna Pavlovna、Soult、Legrand、Merle、Catiche、Thouvenot、Pierre Bezukhov、Andrei Bolkonsky、Marya Bolkonskaya、Vasili Denisov、Boris Drubetskoy、Anatole Kuragin、Hippolyte Kuragin、Platon Karataev、Helene Kuragina、Helene、Kuzma Timokhin、Julie Karagina、Liza、Stepan Nesvitsky、Dunyasha、Vasily Balashev、Annette Scherer、Mavra Kuzminishna、Sergei Wolzogen、Karp Rapp、Pelageya、Fadei Savary、Uvar Friant、Zakhar Berthier、Kutuzov、Agrafena Ivanovna、Bennigsen Jr.、Bennigsen、Lev Raevsky、Tikhon、vasily-balashev、annette-scherer、mavra-kuzminishna、sergei-wolzogen、karp-rapp、fsavary、bennigsen-jr |
